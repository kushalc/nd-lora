#!/usr/bin/env python3
"""
ParScale Local Replication Training Script

Main training script for ParScale experiments on Qwen2.5-0.5B with The Pile dataset.
Supports both PEFT and full fine-tuning modes with comprehensive logging and analysis.
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import modal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils.checkpoint_utils import (load_last_checkpoint, save_last_checkpoint,
                                    sync_logs_to_s3, upload_to_s3)
from utils.contrastive_loss import (ContrastiveLoss, compute_stream_variance,
                                    extract_parscale_stream_logits)
from utils.data_utils import (TokenBudgetTracker, count_tokens_processed,
                              get_tokenizer_info, select_pile_shards,
                              setup_pile_streaming)
from utils.git_utils import check_git_repo_clean
from utils.logging_setup import (log_data_lineage, log_metrics, log_run_header,
                                 log_training_step, setup_python_logging)
from utils.memory_utils import MemoryMonitor
from utils.model_utils import (build_parscale_model,
                               count_trainable_parameters, setup_gpu_training)
from utils.orthogonal_lora_loss import OrthogonalLoRALoss
from utils.stream_diagnostics import StreamDiagnostics
from utils.wandb_setup import (get_git_commit, log_stream_behavior_metrics,
                               log_training_metrics, log_validation_metrics,
                               monitor_system_resources, setup_wandb)

logger = logging.getLogger(__name__)
MODAL_GPU = "A100-80GB"
# MODAL_GPU = "A10G"
# MODAL_GPU = "L4"
MODAL_IMAGE = modal.Image.debian_slim(python_version="3.11") \
                         .pip_install_from_requirements("requirements.txt") \
                         .run_commands("python -m spacy download en_core_web_sm") \
                         .env({"TOKENIZERS_PARALLELISM": "false"}) \
                         .add_local_dir("utils", "/root/utils") \
                         .add_local_dir("configs", "/root/configs") \
                         .add_local_dir("ParScale", "/root/ParScale")
app = modal.App("ParControl")


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ParScale Training Script")

    # Model and training configuration
    parser.add_argument("--training-mode", type=str, default="peft",
                        choices=["peft", "full"], help="Training mode")
    parser.add_argument("--P", type=int, default=1,
                        help="Number of parallel streams")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Base model name from HuggingFace")
    parser.add_argument("--seq-len", type=int, default=976,  # NOTE: 1024 true seq-len, but 48 prefix tokens.
                        help="Maximum sequence length")
    parser.add_argument("--prefix-len", type=int, default=48,
                        help="Prefix length per layer per stream")
    parser.add_argument("--use-stream-lora", action="store_true",
                        help="Use stream-aware LoRA (each stream gets independent LoRA parameters)")

    # Data configuration
    parser.add_argument("--target-tokens", type=int, default=20_000_000,
                        help="Target number of tokens to process")
    parser.add_argument("--scheduler-tokens", type=int, default=100_000_000,
                        help="Number of tokens for LR scheduler (defaults to target-tokens if not specified)")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Micro batch size")
    parser.add_argument("--grad-accumulation", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.02,
                        help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")

    # L-1 Orthogonal LoRA parameters
    parser.add_argument("--orthogonal-lora", action="store_true",
                        help="Enable Orthogonal-LoRA (L-1) experiment")
    parser.add_argument("--lora-layers-start", type=int, default=12,
                        help="Starting layer for LoRA injection")
    parser.add_argument("--lora-layers-end", type=int, default=20,
                        help="Ending layer for LoRA injection")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--lora-modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA. Options: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
    parser.add_argument("--design-layer", type=int, default=16,
                        help="Design layer (l*) for representation decorrelation")
    parser.add_argument("--lambda-bt", type=float, default=0.1,
                        help="Barlow Twins loss weight")
    parser.add_argument("--lambda-perp", type=float, default=0.5,
                        help="Orthogonality penalty weight")
    parser.add_argument("--lambda-kd", type=float, default=0.05,
                        help="Knowledge distillation loss weight")
    parser.add_argument("--bt-method", type=str, default="standard",
                        choices=["mean_vs_others", "standard"],
                        help="Barlow Twins method: mean_vs_others (MvOi) or standard")
    parser.add_argument("--bt-k", type=int,
                        help="Most correlated K streams for which to add Barlow-Twins loss")
    parser.add_argument("--bt-normalization-warmup", action="store_true",
                        help="Enable warmup/normalization in OrthogonalLoRALoss")

    # LoRA ablation parameters
    parser.add_argument("--lora-ablation-enabled", action="store_true",
                        help="Enable LoRA ablation experiments")
    parser.add_argument("--lora-ablation-modules", type=str, nargs="*", default=[],
                        help="LoRA modules to ablate (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)")
    parser.add_argument("--lora-ablation-layers", type=int, nargs="*", default=[],
                        help="Specific layers to ablate (optional)")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name for W&B logging")

    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # System configuration
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run-id", type=str, default=pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S"),
                        help="Run ID of run for use in S3, W&B and local outputs")
    parser.add_argument("--no-sync-to-s3", dest="sync_to_s3", action="store_false", help="Sync to S3")
    parser.add_argument("--s3-base-dir", type=str, default="s3://obviouslywrong-parcontrol/nd-lora",
                        help="Remote directory for checkpoints and logs, generally only used for Colab sessions")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--low-memory-mode", dest="high_memory_mode", action="store_false",
                        help="Enable high-memory optimizations for 10x faster training")
    parser.add_argument("--memory-debug", action="store_true",
                        help="Enable detailed memory diagnostics and OOM prediction")
    parser.add_argument("--compile-model", action="store_true",
                        help="Disable model compilation for speedup (auto-enabled with --high-memory-mode)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of data loading workers (auto-set based on memory mode)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: disable S3 sync and W&B logging to avoid logspam")

    # Contrastive learning parameters
    parser.add_argument("--contrastive-gamma", type=float, default=0.0,
                        help="Contrastive loss weight (0.0 disables contrastive learning)")
    parser.add_argument("--corruption-prob", type=float, default=0.5,
                        help="Probability of corrupting examples for contrastive learning")

    args = parser.parse_args(args=argv)
    args.output_dir = "./outputs/%s" % args.run_id
    args.s3_base_dir = args.s3_base_dir.rstrip("/")

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Override args with config values, ensuring proper type conversion
        for key, value in config_dict.items():
            if hasattr(args, key) and getattr(args, key) is None:
                # Ensure numeric values are properly typed
                if key in ['learning_rate', 'min_lr', 'weight_decay', 'grad_clip', 'warmup_ratio',
                           'contrastive_gamma', 'corruption_prob', 'lambda_bt', 'lambda_perp', 'lambda_kd']:
                    value = float(value)
                elif key in ['P', 'seq_len', 'prefix_len', 'batch_size', 'grad_accumulation',
                             'target_tokens', 'seed', 'target_batch_tokens', 'prefetch_factor',
                             'buffer_size', 'num_workers', 'log_interval', 'eval_interval',
                             'save_interval', 'lora_layers_start', 'lora_layers_end', 'lora_rank', 'design_layer']:
                    value = int(value)
                elif key in ['high_memory_mode', 'wandb_offline', 'memory_debug', 'compile_model']:
                    value = bool(value)
                setattr(args, key, value)

    # High-memory mode only affects data loading strategy, not user parameters
    if args.high_memory_mode:
        logger.info("High-memory mode enabled - will use optimized data loading")
        if args.compile_model is None:
            args.compile_model = True

    # Set minimal defaults only for parameters that truly need them
    if args.num_workers is None:
        args.num_workers = 4 if args.high_memory_mode else 0
    if not hasattr(args, 'prefetch_factor') or args.prefetch_factor is None:
        args.prefetch_factor = 8 if args.high_memory_mode else 2
    if not hasattr(args, 'target_batch_tokens') or args.target_batch_tokens is None:
        args.target_batch_tokens = 65536 if args.high_memory_mode else 8192

    # Handle test mode: disable S3 sync and W&B
    if args.test:
        args.sync_to_s3 = False
        args.wandb_offline = True
        logger.info("Test mode enabled - S3 sync disabled, W&B offline")

    args.git_commit = get_git_commit()
    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic operations (may affect performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_optimizer_scheduler(
    model: nn.Module,
    learning_rate: float,
    min_lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int
) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    # Ensure parameters are the correct type (safety check for YAML loading)
    learning_rate = float(learning_rate)
    min_lr = float(min_lr)
    weight_decay = float(weight_decay)
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)

    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ["bias", "LayerNorm", "layernorm", "layer_norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Linear warmup followed by linear decay to min_lr
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr / learning_rate, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    grad_accumulation_steps: int,
    grad_clip: float,
    step: int,
    contrastive_loss_fn=None,
    orthogonal_lora_loss_fn=None,
    P: int = 1,
    logger=None,
    memory_monitor=None,
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()

    # Memory diagnostics: Before batch loading
    memory_monitor.log_memory("before_batch_load", step)

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Extract contrastive metadata if available
    is_corrupted = batch.get("is_corrupted")
    if is_corrupted is not None:
        is_corrupted = is_corrupted.to(device)

    # Memory diagnostics: Before forward pass
    memory_monitor.log_memory("before_forward", step)

    # Forward pass with mixed precision
    if scaler is not None:
        with torch.autocast(device_type=str(device).split(':')[0]):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_stream_logits=(contrastive_loss_fn is not None),
                output_hidden_states=(orthogonal_lora_loss_fn is not None)
            )

            if orthogonal_lora_loss_fn is not None:
                # L-1 loss computation
                ce_loss = outputs.loss

                # Extract hidden states at design layer
                design_layer_idx = orthogonal_lora_loss_fn.design_layer
                hidden_states = outputs.hidden_states[design_layer_idx]

                # Compute L-1 loss components
                l1_loss, loss_components = orthogonal_lora_loss_fn(step=step, model=model, hidden_states=hidden_states,
                                                                   logits_agg=outputs.logits, logits_backbone=None)

                total_loss = ce_loss + l1_loss
                loss_components["ce_loss"] = ce_loss
                loss_components["total_loss"] = total_loss

            elif contrastive_loss_fn is not None:
                stream_logits = extract_parscale_stream_logits(outputs, P)
                loss, loss_components = contrastive_loss_fn(
                    stream_logits=stream_logits,
                    labels=labels,
                    is_corrupted=is_corrupted,
                    attention_mask=attention_mask
                )
                total_loss = loss
                loss_components["total_loss"] = total_loss

            else:
                total_loss = outputs.loss
                loss_components = {"ce_loss": outputs.loss, "total_loss": total_loss}

            loss = total_loss / grad_accumulation_steps

        # Memory diagnostics: After forward pass
        memory_monitor.log_memory("before_backward", step)

        # Backward pass
        scaler.scale(loss).backward()
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_stream_logits=(contrastive_loss_fn is not None),
            output_hidden_states=(orthogonal_lora_loss_fn is not None)
        )

        if orthogonal_lora_loss_fn is not None:
            # L-1 loss computation
            ce_loss = outputs.loss

            # Extract hidden states at design layer
            design_layer_idx = orthogonal_lora_loss_fn.design_layer
            hidden_states = outputs.hidden_states[design_layer_idx]

            # Compute L-1 loss components
            l1_loss, loss_components = orthogonal_lora_loss_fn(step=step, model=model,
                                                               hidden_states=hidden_states, P=P,
                                                               logits_agg=outputs.logits, logits_backbone=None)

            total_loss = ce_loss + l1_loss
            loss_components["ce_loss"] = ce_loss
            loss_components["total_loss"] = total_loss

        elif contrastive_loss_fn is not None:
            stream_logits = extract_parscale_stream_logits(outputs, P)
            loss, loss_components = contrastive_loss_fn(
                stream_logits=stream_logits,
                labels=labels,
                is_corrupted=is_corrupted,
                attention_mask=attention_mask
            )
            total_loss = loss
            loss_components["total_loss"] = total_loss
        else:
            total_loss = outputs.loss
            loss_components = {"ce_loss": outputs.loss, "total_loss": total_loss}

        loss = total_loss / grad_accumulation_steps

        # Memory diagnostics: After forward pass
        memory_monitor.log_memory("before_backward", step)
        loss.backward()

    # Gradient accumulation and optimization
    memory_monitor.log_memory("before_grad_accumulation", step)
    if (step + 1) % grad_accumulation_steps == 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        optimizer.zero_grad()
    else:
        grad_norm = 0.0

    # Calculate metrics
    batch_size = input_ids.size(0)
    seq_len = attention_mask.sum().item()  # Actual tokens (excluding padding)

    # Build metrics dict including contrastive components
    metrics = {
        "loss": loss.item() * grad_accumulation_steps,
        "perplexity": torch.exp(loss * grad_accumulation_steps).item(),
        "grad_norm": grad_norm,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": optimizer.param_groups[0]["lr"]
    }

    # Add loss components if available
    if loss_components:
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                metrics[f"loss/{key}"] = value.item()
            else:
                metrics[f"loss/{key}"] = value

    return metrics


def validation_step(
    model: nn.Module,
    val_dataloader,
    device: torch.device,
    max_eval_batches: int = 100
) -> Dict[str, float]:
    """Perform validation evaluation."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= max_eval_batches:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            tokens = attention_mask.sum().item()

            total_loss += loss.item() * tokens
            total_tokens += tokens
            num_batches += 1

    if total_tokens == 0:
        return {"loss": float('inf'), "perplexity": float('inf')}

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "eval_batches": num_batches,
        "eval_tokens": total_tokens
    }


@app.function(image=MODAL_IMAGE,
              gpu=MODAL_GPU,
              timeout=3600 * 24,
              volumes={
                  # "/data": modal.Volume.from_name("parcontrol-data", create_if_missing=True),
                  # "/cache": modal.Volume.from_name("parcontrol-cache", create_if_missing=True),
              },
              secrets=[
                  modal.Secret.from_name("wandb"),
                  modal.Secret.from_name("aws"),
              ])
def modal_run_experiment(*nargs, **kwargs):
    return run_experiment(*nargs, **kwargs)


def run_experiment(
    P: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single ParScale experiment for given P value."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Python and W&B logging ASAP
    experiment_name = f"P{P}"
    logger = setup_python_logging(level="INFO", log_dir=config["output_dir"] + "/logs",
                                  experiment_name=experiment_name)
    wandb_run = setup_wandb(config=config, P=P, tokens_M=config["target_tokens"] / 1e6,
                            seq_len=config["seq_len"], seed=config["seed"],
                            run_id=config["run_id"], offline_mode=config.get("wandb_offline", False))
    set_seed(config["seed"])

    # Save code and config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    if config["sync_to_s3"]:
        logging.info("Saving code & config to %s/%s", config["s3_base_dir"], config["run_id"])
        upload_to_s3(Path.cwd(), config["s3_base_dir"] + f"/{config['run_id']}", logger=logger)
        upload_to_s3(config_path, config["s3_base_dir"], config["output_dir"], logger)

    # Modal-specific setup
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/cache/datasets"

    # Ensure cache directories exist
    for cache_dir in ["/cache/huggingface", "/cache/transformers", "/cache/datasets"]:
        os.makedirs(cache_dir, exist_ok=True)

    # Setup memory monitor
    memory_monitor = MemoryMonitor(logger, detailed=config.get("memory_debug", False))

    # Log run header
    log_run_header(logger, config)

    # Setup device and mixed precision
    device, scaler = setup_gpu_training(high_memory_mode=config.get("high_memory_mode", False),
                                        enable_mixed_precision=True)
    config["device"] = device
    logger.info(f"Using device: {device}")

    # Load base model and tokenizer
    logger.info("Loading base model and tokenizer...")
    memory_monitor.log_memory("before_model_load", -1)
    model, tokenizer = build_parscale_model(config)

    # Compile model for speedup (auto-enabled with high-memory-mode or explicit flag)
    should_compile = config.get("compile_model", False)
    if should_compile:
        logger.info("Compiling model for speedup...")
        try:
            # Use torch.compile with aggressive optimizations
            model = torch.compile(
                model,
                mode="max-autotune",  # Aggressive optimization for A100
                fullgraph=False,      # Allow graph breaks for compatibility with PEFT
                dynamic=True          # Handle dynamic shapes from variable seq lengths
            )
            logger.info("Model compilation successful - expect 10-20% speedup")
        except Exception as e:
            logger.warning("Model compilation failed, continuing without compilation", exc_info=True)

    # Log parameter counts
    param_info = count_trainable_parameters(model)
    logger.info("Parameter counts:")
    for key, value in param_info.items():
        logger.info(f"  {key}: {value:,}")

    # Setup data
    logger.info("Setting up data streams...")
    memory_monitor.log_memory("before_data_setup", -1)
    train_shard_ids, val_shard_ids = select_pile_shards(1)  # Note: Using single shard for training

    # if config.get("high_memory_mode", False):
    #     logger.info("Using high-performance data loading...")
    #     train_dataloader, val_dataloader = create_high_performance_dataloader(
    #         tokenizer=tokenizer,
    #         max_seq_len=config["seq_len"],
    #         batch_size=config["batch_size"],
    #         num_workers=config.get("num_workers", 4),
    #         seed=config["seed"],
    #         prefetch_factor=config.get("prefetch_factor", 8),
    #         target_batch_tokens=config.get("target_batch_tokens", 65536),
    #         pack_sequences=True,
    #         buffer_size=20000 if config.get("high_memory_mode", False) else 10000  # Conservative buffer for 15GB
    #     )
    # else:
    # logger.info("Using standard data loading...")

    # Setup optimizer, scheduler & token tracker
    memory_monitor.log_memory("before_optimizer_setup", -1)

    # Use scheduler_tokens if specified, otherwise use target_tokens
    scheduler_tokens = config.get("scheduler_tokens") or config["target_tokens"]
    scheduler_steps = count_tokens_processed(config["batch_size"], scheduler_tokens, config["seq_len"],
                                             config["grad_accumulation"])
    warmup_steps = int(scheduler_steps * config["warmup_ratio"])
    actual_steps = count_tokens_processed(config["batch_size"], config["target_tokens"], config["seq_len"],
                                          config["grad_accumulation"])

    optimizer, scheduler = setup_optimizer_scheduler(
        model=model,
        learning_rate=config["learning_rate"],
        min_lr=config["min_lr"],
        weight_decay=config["weight_decay"],
        total_steps=scheduler_steps,
        warmup_steps=warmup_steps
    )

    logger.info(f"Training plan: target_tokens={config['target_tokens']:,} scheduler_tokens={scheduler_tokens:,} "
                f"actual_steps={actual_steps:,} scheduler_steps={scheduler_steps:,} warmup_steps={warmup_steps:,}")
    token_tracker = TokenBudgetTracker(target_tokens=config["target_tokens"], seq_len=config["seq_len"],
                                       batch_size=config["batch_size"])

    # Try to resume from last checkpoint
    resume_step = load_last_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                       token_tracker=token_tracker, P=P, config=config, logger=logger)

    # Setup dataloaders (contrastive vs standard)
    train_examples_skip = val_examples_skip = 0
    if resume_step:
        train_examples_skip = resume_step * config["batch_size"]
        val_examples_skip = (resume_step // config["eval_interval"]) * 100 * config["batch_size"]  # assumes default from max_eval_batches

    # Setup contrastive loss if enabled
    contrastive_loss_fn = None
    if config.get("contrastive_gamma", 0.0) > 0.0:
        logger.info("Enabling contrastive learning with gamma=%.3f", config['contrastive_gamma'])
        contrastive_loss_fn = ContrastiveLoss(gamma=config["contrastive_gamma"])

    # Setup L-1 orthogonal LoRA loss if enabled
    orthogonal_lora_loss_fn = None
    if config.get("orthogonal_lora", False):
        kwargs = dict(warmup_steps=warmup_steps, design_layer=config['design_layer'],
                      lambda_bt=config['lambda_bt'], bt_method=config['bt_method'],
                      bt_normalization_warmup=config['bt_normalization_warmup'],
                      bt_k=config["bt_k"])
        logger.info("Enabling OrthogonalLoRALoss with parameters: %s", kwargs)
        orthogonal_lora_loss_fn = OrthogonalLoRALoss(P=P, **kwargs)

    train_dataloader = setup_pile_streaming(
        tokenizer=tokenizer,
        shard_ids=train_shard_ids,
        seq_len=config["seq_len"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        skip_examples=train_examples_skip,
        corruption=contrastive_loss_fn is not None,
    )
    val_dataloader = setup_pile_streaming(
        tokenizer=tokenizer,
        shard_ids=val_shard_ids,
        seq_len=config["seq_len"],
        batch_size=config["batch_size"],
        skip_examples=val_examples_skip,
        corruption=contrastive_loss_fn is not None,
    )

    # Log data lineage
    tokenizer_info = get_tokenizer_info(tokenizer)
    log_data_lineage(logger, train_shard_ids, tokenizer_info, config["seed"])

    # Setup stream diagnostics
    stream_diagnostics = StreamDiagnostics(P)

    # Training loop
    logger.info("Starting training...")
    abs_start_time = start_time = time.time()
    step = resume_step if resume_step is not None else 0
    best_val_loss = float('inf')
    memory_monitor.log_and_clear_memory("before_start", step)

    try:
        for batch in train_dataloader:
            # Training step
            start_time = time.time()
            step_metrics = training_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                grad_accumulation_steps=config["grad_accumulation"],
                grad_clip=config["grad_clip"],
                step=step,
                contrastive_loss_fn=contrastive_loss_fn,
                orthogonal_lora_loss_fn=orthogonal_lora_loss_fn,
                P=P,
                logger=logger,
                memory_monitor=memory_monitor,
            )

            # Update learning rate
            if (step + 1) % config["grad_accumulation"] == 0:
                scheduler.step()

            # Update token tracker
            tokens_complete = token_tracker.update(step_metrics["batch_size"])
            step_metrics.update(token_tracker.get_progress())

            # Add timing information
            step_metrics["step_time"] = time.time() - start_time
            step_metrics["total_time"] = time.time() - abs_start_time
            start_time = time.time()

            # Log training metrics
            if step % config["log_interval"] == 0:
                log_training_step(logger, step, step_metrics, config["log_interval"])

                # Monitor system resources
                system_stats = monitor_system_resources()

                # Extract stream diagnostics (periodically to avoid overhead)
                stream_stats = {}
                # Note: Stream diagnostics disabled on MPS due to compatibility issues
                # if step % (config["log_interval"] * 2) == 0:
                #     try:
                #         stream_weights = stream_diagnostics.extract_stream_weights(
                #             model, batch["input_ids"], batch["attention_mask"]
                #         )
                #         stream_stats = stream_diagnostics.compute_stream_statistics(
                #             stream_weights, batch["attention_mask"]
                #         )
                #         stream_diagnostics.record_statistics(step, stream_stats)
                #     except Exception as e:
                #         logger.warning("Stream diagnostics failed: %s", e, exc_info=True)

                # Log memory summary to W&B
                memory_summary = memory_monitor.get_memory_summary()
                if memory_summary and wandb_run:
                    wandb_run.log({
                        "memory/current_allocated_gb": memory_summary['current_allocated_gb'],
                        "memory/peak_allocated_gb": memory_summary['peak_allocated_gb'],
                        "memory/current_usage_percent": memory_summary['current_usage_percent'],
                        "memory/peak_usage_percent": memory_summary['peak_usage_percent'],
                        "memory/oom_warnings_count": memory_summary['oom_warnings_count'],
                        "memory/trend": memory_summary['memory_trend']
                    }, step=step)

                # Compute contrastive behavior metrics if enabled
                contrastive_stats = {}
                if contrastive_loss_fn is not None:
                    with torch.no_grad():
                        outputs = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device),
                            return_stream_logits=True
                        )
                        stream_logits = extract_parscale_stream_logits(outputs, P)
                        if len(stream_logits) > 1:
                            stream_disagreement = compute_stream_variance(stream_logits)
                            contrastive_stats = log_stream_behavior_metrics(
                                stream_disagreement=stream_disagreement,
                                is_corrupted=batch["is_corrupted"].to(device),
                                step=step,
                                corruption_types=batch.get("corruption_types")
                            )

                # Log to W&B, don't commit if validation is about to happen at the same step
                log_training_metrics(step, step_metrics, loss_components=None, stream_stats=stream_stats,
                                     system_stats=system_stats, contrastive_stats=contrastive_stats or {}, logger=logger,
                                     commit=step % config["eval_interval"] != 0)

            if step % config["save_interval"] == 0 and step > 0:
                # Save versioned checkpoint based on step
                versioned_basename = f"model_state_{step:05d}.pt"
                save_last_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                     step=step, config=config, token_tracker=token_tracker,
                                     P=P, logger=logger, basename=versioned_basename)

            # Validation
            if step % config["eval_interval"] == 0 and step > 0:
                memory_monitor.log_and_clear_memory("before_validation", step)

                val_metrics = validation_step(model, val_dataloader, device)
                log_metrics(logger, step, val_metrics, name="validation_step")
                log_validation_metrics(step, val_metrics, logger)

                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    addl_data = {
                        "val_loss": val_metrics["loss"],
                    }
                    save_last_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                         step=step, config=config, token_tracker=token_tracker,
                                         P=P, logger=logger, basename="best_model_state.pt",
                                         addl_data=addl_data)
                    best_val_loss = val_metrics["loss"]

                # Clear cache after validation
                memory_monitor.log_and_clear_memory("before_sync", step)

                # Upload logs to S3 if requested
                if config.get("sync_to_s3", False):
                    log_dir = str(Path(config["output_dir"]) / "logs")
                    sync_logs_to_s3(log_dir, config["s3_base_dir"], config["output_dir"], logger=logger)

            # Check if target tokens reached
            if tokens_complete:
                logger.info("Target token budget reached!")
                break

            step += 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save final checkpoint on interruption
        if step > 0:
            save_last_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                 step=step, config=config, token_tracker=token_tracker,
                                 P=P, logger=logger)

    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)

        # Save checkpoint on failure for debugging
        if step > 0:
            try:
                save_last_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                     step=step, config=config, token_tracker=token_tracker,
                                     P=P, logger=logger)
            except Exception as checkpoint_err:
                logger.error("Failed to save emergency checkpoint", exc_info=True)

        # Final memory diagnostics for debugging OOM
        final_memory_summary = memory_monitor.get_memory_summary()
        logger.error(f"Final memory state: {final_memory_summary}")
        logger.error(f"Total OOM warnings during training: {memory_monitor.oom_warnings}")

        # Check if this was an OOM error
        if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
            logger.error("CONFIRMED OOM ERROR - Memory diagnostics:")
            logger.error(f"  Peak memory usage: {final_memory_summary.get('peak_usage_percent', 0):.1f}%")
            logger.error(f"  Memory trend before OOM: {final_memory_summary.get('memory_trend', 0)*100:.2f}% per step")
            logger.error("  Suggested fixes: Reduce batch_size, seq_len, or enable gradient_checkpointing")

        raise

    # Save final checkpoint
    save_last_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                         step=step, config=config, token_tracker=token_tracker,
                         P=P, logger=logger)

    # Note: Stream diagnostics functionality has been moved to separate utilities

    # Calculate total time and throughput
    total_time = time.time() - start_time
    tokens_per_second = token_tracker.processed_tokens / total_time if total_time > 0 else 0

    # Final metrics
    final_metrics = {
        # **final_val_metrics,
        # **final_stream_stats,
        "total_steps": step,
        "total_tokens": token_tracker.processed_tokens,
        "total_time_seconds": total_time,
        "tokens_per_second": tokens_per_second,
        "best_val_loss": best_val_loss
    }
    log_metrics(logger, step, final_metrics, name="final_result")
    wandb_run.finish()

    # Save results
    results_path = Path(config["output_dir"]) / f"results_P{P}.json"
    with open(results_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Final sync to S3 if requested
    if config.get("sync_to_s3", False):
        log_dir = str(Path(config["output_dir"]) / "logs")
        sync_logs_to_s3(log_dir, config["s3_base_dir"], config["output_dir"], logger=logger)

        # Upload diagnostics if they exist
        diagnostics_dir = Path(config["output_dir"]) / "diagnostics"
        if diagnostics_dir.exists():
            upload_to_s3(str(diagnostics_dir), config["s3_base_dir"], config["output_dir"], logger)

        # Upload results files
        results_files = list(Path(config["output_dir"]).glob("results_*.json"))
        for result_file in results_files:
            upload_to_s3(str(result_file), config["s3_base_dir"], config["output_dir"], logger)

        # Upload config file
        config_file = Path(config["output_dir"]) / "config.yaml"
        if config_file.exists():
            upload_to_s3(str(config_file), config["s3_base_dir"], config["output_dir"], logger)

    # logger.info("Experiment completed successfully! Results [loss=%.4f, perplexity=%.1e, ...] saved to: %s",
    #             final_metrics['loss'], final_metrics['perplexity'], results_path)
    return final_metrics


def main(argv=None):
    args = parse_args(argv)
    check_git_repo_clean()
    modal_run_experiment.remote(args.P, vars(args))




# == comparable set for P=1
@app.local_entrypoint()
def modal__P1__r32():
    main([
        "--P=1",
        "--lora-rank=32",
    ])


@app.local_entrypoint()
def modal__P1__r64():
    main([
        "--P=1",
        "--lora-rank=64",
    ])


@app.local_entrypoint()
def modal__P1__r128():
    main([
        "--P=1",
        "--lora-rank=128",
    ])


# == comparable set for P=2
@app.local_entrypoint()
def modal__P2__r32():
    main([
        "--P=2",
        "--lora-rank=32",
    ])


# == comparable set for P=4
@app.local_entrypoint()
def modal__P4__r64():
    main([
        "--P=4",
        "--lora-rank=64",
    ])


@app.local_entrypoint()
def modal__lP4__r64():  # ParScale-BT ablation
    main([
        "--P=4",
        "--lora-rank=64",
        "--orthogonal-lora",
    ])


@app.local_entrypoint()
def modal__slP4():
    main([
        "--P=4",
        "--use-stream-lora",
        "--orthogonal-lora",
    ])


@app.local_entrypoint()
def modal__nslP4():
    main([
        "--P=4",
        "--use-stream-lora",
        "--orthogonal-lora",
        "--bt-normalization-warmup",
    ])


# == OptC9 configurations for Qwen2.5-0.5B (Optuna-optimized hyperparameters)
@app.local_entrypoint()
def modal__nslP2__OptC9():
    """ND-LoRA P=2 with Optuna-optimized hyperparameters (trial 009)"""
    main([
        "--P=2",
        "--use-stream-lora",
        "--orthogonal-lora",
        "--bt-normalization-warmup",
        "--design-layer=20",
        "--lambda-bt=0.2899044045624172",
        "--lora-modules", "q_proj", "k_proj", "v_proj",
        "--lora-rank=16",
    ])


@app.local_entrypoint()
def modal__nslP4__OptC9():
    """ND-LoRA P=4 with Optuna-optimized hyperparameters (trial 001)"""
    main([
        "--P=4",
        "--use-stream-lora",
        "--orthogonal-lora",
        "--bt-normalization-warmup",
        "--design-layer=20",
        "--lambda-bt=0.5779870909936439",
        "--lora-modules", "q_proj", "k_proj", "v_proj",
        "--lora-rank=16",
    ])


@app.local_entrypoint()
def modal__nslP8__OptC9():
    """ND-LoRA P=8 with Optuna-optimized hyperparameters (trial 002)"""
    main([
        "--P=8",
        "--use-stream-lora",
        "--orthogonal-lora",
        "--bt-normalization-warmup",
        "--design-layer=20",
        "--lambda-bt=0.12656544871427247",
        "--lora-modules", "q_proj", "k_proj", "v_proj",
        "--lora-rank=16",
    ])


@app.local_entrypoint()
def modal__sP4():
    """Stream-LoRA P=4 without orthogonal constraint"""
    main([
        "--P=4",
        "--use-stream-lora",
        "--lora-rank=16",
    ])


# == comparable set for P=8
@app.local_entrypoint()
def modal__P8__r128():
    main([
        "--P=8",
        "--lora-rank=128",
    ])


# == LoRA Ablation Experiments ==
@app.local_entrypoint()
def modal__p4_nOSL_ablation__modules():
    check_git_repo_clean()

    ablation_configs = [
        {"modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "name": "no_attention"},
        {"modules": ["gate_proj", "up_proj", "down_proj"], "name": "no_mlp"},
    ]

    for config in ablation_configs:
        args = [
            "--P=4",
            "--use-stream-lora",
            "--orthogonal-lora",
            "--bt-normalization-warmup",
            "--lora-ablation-enabled",
            "--name", config["name"]
        ] + ["--lora-ablation-modules"] + config["modules"]

        parsed_args = parse_args(args)
        run_experiment.spawn(parsed_args.P, vars(parsed_args))
        print(f"Spawned ablation run: {config['name']}")
        print(f"Sleeping for some time before spawning next run to generate unique id..")
        time.sleep(10)

    print(f"Spawned {len(ablation_configs)} module ablation experiments")
