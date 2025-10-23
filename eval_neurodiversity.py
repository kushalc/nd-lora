#!/usr/bin/env python3
"""
Neurodiversity evaluation using PyTorch native hooks for activation recording.

Supports both local and Modal execution for GPU-accelerated evaluation.
Uses PyTorch's register_forward_hook API instead of nnsight for reliability.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import modal
import numpy as np
import pandas as pd
import torch

from leaderboard.backend_cli import get_model_alias
from leaderboard.src.backend.manage_requests import EvalRequest
from leaderboard.src.backend.run_eval_suite import run_evaluation
from leaderboard.src.utils import get_tasks_by_benchmarks
from utils.checkpoint_utils import check_and_download_from_s3, upload_to_s3
from utils.stream_aware_lora import parse_streams_from_batch

MODAL_GPU = "A10G"  # A10G for better memory and performance
MODAL_IMAGE = modal.Image.debian_slim(python_version="3.10") \
    .pip_install_from_requirements(Path(__file__).parent / "requirements.txt") \
    .pip_install_from_requirements(Path(__file__).parent / "leaderboard" / "requirements.txt") \
    .run_commands("python -m spacy download en_core_web_sm") \
    .env({"TOKENIZERS_PARALLELISM": "false"}) \
    .add_local_dir(Path(__file__).parent / "leaderboard", "/root/leaderboard") \
    .add_local_dir(Path(__file__).parent / "leaderboard/src", "/root/src") \
    .add_local_dir(Path(__file__).parent / "utils", "/root/utils") \
    .add_local_dir(Path(__file__).parent / "ParScale", "/root/ParScale")
app = modal.App("ParControl-Causality")


def get_module_by_name(model, module_name: str):
    """Navigate to a module using dot notation (e.g., 'model.layers.0.mlp')."""
    module = model
    for part in module_name.split('.'):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

# Compute cross-correlations more efficiently
# For Definition 1: Cij = E[z˜i z˜j⊤], ||Cij||²_F = (z˜i⊤ z˜j)²


def compute_dspec(weights_flat) -> float:
    """
    Compute cross-correlation between streams using proper whitening.

    Input shape: [batch, sequence, hidden_dim, P]
    Whitening: across batch and sequence dimensions to remove input-specific correlations
    Correlation: cosine similarity between streams in hidden dimension space
    """
    P, batch, seq_len, hidden_dim = weights_flat.shape

    # Reshape to combine batch and sequence dimensions for whitening
    # Shape: [batch*seq_len, hidden_dim, P]
    weights_reshaped = weights_flat.view(P, -1, hidden_dim)

    # Whiten across the batch*sequence dimension (dim=0)
    # This removes correlations from different inputs and positions
    weights_mean = weights_reshaped.mean(dim=1, keepdim=True)  # [1, hidden_dim, P]
    weights_centered = weights_reshaped - weights_mean

    # Compute std across batch*sequence for each hidden dim and stream
    weights_std = weights_centered.std(dim=1, keepdim=True)  # [1, hidden_dim, P]
    weights_whitened = weights_centered / (weights_std + 1e-8)

    d_spec = 0.0
    count = 0
    for i in range(P):
        for j in range(i+1, P):
            # Extract stream vectors: [batch*seq_len, hidden_dim]
            stream_i_flat = weights_whitened[i, :, :].flatten()
            stream_j_flat = weights_whitened[j, :, :].flatten()

            cosine_sim = torch.nn.functional.cosine_similarity(stream_i_flat.unsqueeze(0),
                                                               stream_j_flat.unsqueeze(0), dim=1)
            d_spec += cosine_sim.item()
            count += 1

    # Average correlation across all stream pairs
    d_spec = d_spec / count if count > 0 else 0.0
    return d_spec


def substitute_tokens(reshaped_output: torch.Tensor, k_substitute: int,
                      rng: np.random.RandomState) -> torch.Tensor:
    """
    Vectorized stream substitution corruption: randomly substitute k streams with copies of other streams.

    Args:
        reshaped_output: Tensor of shape [P, batch, seq_len, hidden_dim]
        k_substitute: Number of streams to substitute per token position
        rng: NumPy random state for reproducibility

    Returns:
        Corrupted tensor with same shape as input
    """
    P, batch_size, seq_len, hidden_dim = reshaped_output.shape
    assert 0 <= k_substitute < P, f"k_substitute must be in (0, {P}), got {k_substitute}"
    if k_substitute == 0:
        return reshaped_output

    corrupted = reshaped_output.clone()
    n_positions = batch_size * seq_len

    # Vectorized victim selection using Gumbel-max trick: [n_positions, k_substitute]
    gumbel_noise = -np.log(-np.log(rng.uniform(0, 1, size=(n_positions, P)) + 1e-20) + 1e-20)
    victim_indices = np.argsort(gumbel_noise, axis=1)[:, :k_substitute]

    # Vectorized donor selection using masked random sampling
    # Generate random priorities for all possible donors: [n_positions, k_substitute, P]
    donor_priorities = rng.uniform(0, 1, size=(n_positions, k_substitute, P))

    # Mask out victims from donor pool by setting their priority to -inf
    victim_mask = np.zeros((n_positions, k_substitute, P), dtype=bool)
    victim_mask[np.arange(n_positions)[:, None], np.arange(k_substitute)[None, :], victim_indices] = True
    donor_priorities[victim_mask] = -np.inf

    # Select donor with highest priority (excludes victim automatically)
    donor_indices = np.argmax(donor_priorities, axis=2)  # [n_positions, k_substitute]

    # Reshape for tensor indexing: [batch, seq_len, k_substitute]
    victim_indices = torch.from_numpy(victim_indices).view(batch_size, seq_len, k_substitute)
    donor_indices = torch.from_numpy(donor_indices).view(batch_size, seq_len, k_substitute)

    # Create advanced indexing arrays
    batch_idx = torch.arange(batch_size, device=corrupted.device)[:, None, None].expand(
        batch_size, seq_len, k_substitute)
    seq_idx = torch.arange(seq_len, device=corrupted.device)[None, :, None].expand(
        batch_size, seq_len, k_substitute)

    # Move indices to device
    victim_indices = victim_indices.to(corrupted.device)
    donor_indices = donor_indices.to(corrupted.device)

    # Reshape corrupted from [P, batch, seq, hidden] to [batch, seq, P, hidden] for indexing
    corrupted_reordered = corrupted.permute(1, 2, 0, 3)  # [batch, seq, P, hidden]
    reshaped_reordered = reshaped_output.permute(1, 2, 0, 3)  # [batch, seq, P, hidden]

    # Vectorized substitution: corrupted[b, t, victim] = streams[b, t, donor]
    corrupted_reordered[batch_idx, seq_idx, victim_indices] = reshaped_reordered[batch_idx, seq_idx, donor_indices]

    # Reshape back to [P, batch, seq, hidden]
    corrupted = corrupted_reordered.permute(2, 0, 1, 3)

    return corrupted


def substitute_streams(reshaped_output: torch.Tensor, k_substitute: int,
                       fraction: float, rng: np.random.RandomState) -> torch.Tensor:
    """
    Per-sample consistent stream substitution with fractional token replacement.

    For corruption mode "stream":
    - Pick k_substitute < P victim streams (same for all tokens in a sample)
    - For each victim, pick donor from non-victim streams
    - For each victim, substitute fraction% of tokens with donor's hidden states
    - Designed for causal dose-response analysis

    Args:
        reshaped_output: Tensor of shape [P, batch, seq_len, hidden_dim]
        k_substitute: Number of victim streams (must be < P)
        fraction: Fraction of tokens to substitute (0.0 to 1.0)
        rng: NumPy random state for reproducibility

    Returns:
        Corrupted tensor with same shape as input
    """
    P, batch_size, seq_len, hidden_dim = reshaped_output.shape
    assert 0 <= k_substitute < P, f"k_substitute must be in (0, {P}), got {k_substitute}"
    assert 0.0 <= fraction <= 1.0, f"fraction must be in [0, 1], got {fraction}"
    if k_substitute == 0 or fraction == 0:
        return reshaped_output

    corrupted = reshaped_output.clone()
    n_victim_tokens = int(fraction * seq_len)

    for b in range(batch_size):
        # (1) Pick k_substitute victims for this sample
        victims = rng.choice(P, size=k_substitute, replace=False)
        available_donors = [i for i in range(P) if i not in victims]
        assert len(available_donors) > 0, f"No available donors for batch {b}"

        # (2) For each victim, pick donor from available streams and substitute
        for victim in victims:
            donor = rng.choice(available_donors)
            victim_token_pos = rng.choice(seq_len, size=n_victim_tokens, replace=False)
            corrupted[victim, b, victim_token_pos, :] = reshaped_output[donor, b, victim_token_pos, :]

    return corrupted


def build_instrumented_eval_request(model_name: str, target_layer: str, activation_storage: List,
                                    corruption_mode: str = "none",
                                    substitute_k_streams: int = 1,
                                    substitute_fraction: float = 1.0,
                                    model_builder=None):
    """Create EvalRequest and add hooks to capture activations and optionally corrupt streams.

    Args:
        model_name: Model identifier or S3 path
        target_layer: Layer to instrument
        activation_storage: List to store activation results
        corruption_mode: Corruption method (none, alpha, token, stream)
        substitute_k_streams: Number of victim streams for corruption
        substitute_fraction: Fraction of tokens to corrupt
        model_builder: Optional custom model builder for S3 models (e.g., for IHD models)
    """
    # Create standard EvalRequest and load model
    eval_request = EvalRequest(model=model_name, private=False, status="RUNNING", json_filepath="")
    eval_request.load_model(model_builder=model_builder)
    assert eval_request._model is not None, f"Failed to load model: {model_name}"

    # Detect IHD models (no ParScale attributes) and skip instrumentation
    if not hasattr(eval_request._model.model.model, 'parscale_n'):
        logging.info("Detected IHD model (no parscale_n attribute) - skipping instrumentation")
        return eval_request

    parscale_n = eval_request._model.model.model.parscale_n
    hidden_size = eval_request._model.model.model.aggregate_layer[0].out_features

    # Validate corruption parameters
    if corruption_mode not in ["none", "alpha", "token", "stream"]:
        raise ValueError(f"Invalid corruption_mode: {corruption_mode}")

    if corruption_mode in ["token", "stream"]:
        assert 0 <= substitute_k_streams < parscale_n, \
            f"substitute_k_streams must be in (0, {parscale_n}), got {substitute_k_streams}"
        if corruption_mode == "stream":
            assert 0.0 <= substitute_fraction <= 1.0, \
                f"substitute_fraction must be in [0, 1], got {substitute_fraction}"
            logging.info("Corruption mode: stream (k=%s, fraction=%.3f, P=%s)",
                         substitute_k_streams, substitute_fraction, parscale_n)
        else:  # token mode
            logging.info("Corruption mode: token (k=%s, P=%s)", substitute_k_streams, parscale_n)
    elif corruption_mode == "alpha":
        logging.info("Corruption mode: alpha")
    else:
        logging.info("Corruption mode: none")

    def make_combined_hook(name):
        """Combined hook that does both neurodiversity monitoring and optional corruption."""
        def hook_fn(module, inputs, output):
            if getattr(module, "_active_hook", False):
                return output
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            assert inputs.shape[0] % parscale_n == 0, f"Batch size {inputs.shape[0]} not divisible by P={parscale_n}"
            assert inputs.shape[-1] == hidden_size, f"Hidden size mismatch: got {inputs.shape[-1]}, expected {hidden_size}"
            assert eval_request._alpha is not None

            try:
                module._active_hook = True
                reshaped_output = parse_streams_from_batch(output, parscale_n)
                results = {
                    "original_dspec": compute_dspec(reshaped_output),
                    "corruption_mode": corruption_mode,
                }
                activation_storage.append(results)

                if corruption_mode == "none":
                    logging.debug("No corruption applied")
                    return output

                # Apply corruption based on mode
                if corruption_mode == "alpha":
                    # Alpha-blending corruption
                    mean_output = reshaped_output.mean(dim=0, keepdim=True)
                    alpha = eval_request._alpha
                    corrupted_output = reshaped_output * (1 - alpha) + mean_output * alpha

                    results.update({
                        'alpha': alpha,
                    })
                    logging.debug("Applied alpha-blending corruption: alpha=%.6f", alpha)

                elif corruption_mode == "token":
                    # Per-token random stream substitution; not updating results with metadata because always same.
                    rng = eval_request._corruption_rng
                    corrupted_output = substitute_tokens(reshaped_output, substitute_k_streams, rng)
                    logging.debug("Applied per-token stream substitution: k=%s", substitute_k_streams)

                elif corruption_mode == "stream":
                    # Per-sample fractional stream substitution; not updating results metadata because always same.
                    rng = eval_request._corruption_rng
                    corrupted_output = substitute_streams(reshaped_output, substitute_k_streams, substitute_fraction, rng)
                    logging.debug("Applied fractional stream substitution: k=%s fraction=%.3f",
                                  substitute_k_streams, substitute_fraction)

                results["corrupted_dspec"] = compute_dspec(corrupted_output)
                corrupted_output = corrupted_output.view(output.shape)
                return corrupted_output

            finally:
                module._active_hook = False

        return hook_fn

    module = get_module_by_name(eval_request._model, target_layer)
    combined_hook = make_combined_hook(target_layer)
    eval_request._activation_hooks = [
        module.register_forward_hook(combined_hook)
    ]
    corruption_details = {
        "none": "none",
        "alpha": "alpha_blending",
        "token": f"token_substitution(k={substitute_k_streams})",
        "stream": f"stream_substitution_fractional(k={substitute_k_streams}, f={substitute_fraction})"
    }[corruption_mode]
    logging.info("Registered combined hook for layer: %s (corruption_mode=%s details=%s)",
                 target_layer, corruption_mode, corruption_details)
    return eval_request


def build_instrumented_eval_request_with_module_groups(
    model_name: str,
    layer_idx: int,
    module_types: List[str],
    activation_storage: Dict[str, List]
):
    """Create EvalRequest with hooks for multiple modules within a single layer.

    Args:
        model_name: Model identifier or S3 path
        layer_idx: Layer index to hook (e.g., 5 for layer 5)
        module_types: List of module types to hook from ["self_attn", "mlp", "layer"]
        activation_storage: Dict mapping module_type -> List of activation results

    Returns:
        EvalRequest with registered hooks for each module type
    """
    # Create standard EvalRequest and load model
    eval_request = EvalRequest(model=model_name, private=False, status="RUNNING", json_filepath="")
    eval_request.load_model()
    assert eval_request._model is not None, f"Failed to load model: {model_name}"

    # Detect IHD models (no ParScale attributes) and skip instrumentation
    if not hasattr(eval_request._model.model.model, 'parscale_n'):
        logging.info("Detected IHD model (no parscale_n attribute) - skipping module group instrumentation")
        return eval_request

    parscale_n = eval_request._model.model.model.parscale_n
    hidden_size = eval_request._model.model.model.aggregate_layer[0].out_features

    # Validate inputs
    assert isinstance(module_types, list) and len(module_types) > 0
    assert all(mt in ["self_attn", "mlp", "layer"] for mt in module_types), \
        f"module_types must be 'self_attn', 'mlp', or 'layer', got {module_types}"
    assert isinstance(activation_storage, dict)
    assert all(mt in activation_storage for mt in module_types), \
        f"activation_storage must have keys for all module_types"

    # Build module paths for the specified layer
    target_module_paths = []
    for mt in module_types:
        if mt == "layer":
            # Hook entire decoder layer (baseline measurement)
            target_module_paths.append(f"model.model.layers.{layer_idx}")
        else:
            # Hook specific module (self_attn or mlp)
            target_module_paths.append(f"model.model.layers.{layer_idx}.{mt}")

    logging.info("Registering %d hooks for layer %d (modules=%s)",
                 len(target_module_paths), layer_idx, module_types)

    def make_module_hook(module_path: str):
        """Create hook for multi-module mode (no corruption support)."""
        parts = module_path.split(".")
        assert len(parts) in [4, 5], f"Expected 'model.model.layers.X[.module]', got {module_path}"
        hook_layer_idx = int(parts[3])

        if len(parts) == 4:
            # Entire decoder layer: model.model.layers.5
            module_type = "layer"
        else:
            # Specific module: model.model.layers.5.self_attn
            module_type = parts[4]

        assert module_type in activation_storage, \
            f"module_type '{module_type}' not in storage: {list(activation_storage.keys())}"

        def hook_fn(module, inputs, output):
            if getattr(module, "_active_hook", False):
                return output

            try:
                module._active_hook = True

                # Extract hidden_states from tuple output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Safety checks on output
                assert hidden_states.shape[0] % parscale_n == 0, \
                    f"Batch size {hidden_states.shape[0]} not divisible by P={parscale_n}"
                assert hidden_states.shape[-1] == hidden_size, \
                    f"Hidden size {hidden_states.shape[-1]} != expected {hidden_size}"

                # Compute dspec
                reshaped_output = parse_streams_from_batch(hidden_states, parscale_n)
                dspec_value = compute_dspec(reshaped_output)

                # Store results in dict by module type
                results = {
                    "original_dspec": dspec_value,
                    "layer_idx": hook_layer_idx,
                    "module_type": module_type,
                }
                activation_storage[module_type].append(results)

                return output

            finally:
                module._active_hook = False

        return hook_fn

    # Register hooks for each module
    eval_request._activation_hooks = []

    for module_path in target_module_paths:
        module = get_module_by_name(eval_request._model, module_path)
        hook = make_module_hook(module_path)
        eval_request._activation_hooks.append(
            module.register_forward_hook(hook)
        )

    logging.info("Successfully registered %d hook(s)", len(eval_request._activation_hooks))
    return eval_request


def run_instrumented_evaluation(model_names: List[str], target_layer: str, tasks: List[str],
                                output_dir: Path, resamplings: int = 16, limit: int = 128,
                                s3_base_dir: Optional[str] = None, force: bool = False,
                                corruption_mode: str = "none", corruption_alpha_max: float = 1.0,
                                substitute_k_streams: int = 1, substitute_fraction: float = 1.0,
                                seed: int = 42, model_builder=None) -> Dict:
    now = pd.Timestamp.now(tz="US/Pacific")
    log_path = output_dir / now.strftime("eval_neurodiversity.%Y-%m-%d-%H-%M-%S.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info("Setting system-wide psuedo-RNG seed to %d", seed)
    np.random.seed(seed)
    random.seed(seed)

    results = {}
    task_lst = get_tasks_by_benchmarks(tasks)
    rng = np.random.default_rng(seed=seed)
    try:
        for model_name in random.sample(model_names, len(model_names)):
            activation_storage = []
            eval_request = build_instrumented_eval_request(
                model_name=model_name,
                target_layer=target_layer,
                activation_storage=activation_storage,
                corruption_mode=corruption_mode,
                substitute_k_streams=substitute_k_streams,
                substitute_fraction=substitute_fraction,
                model_builder=model_builder
            )

            for task in random.sample(task_lst, len(task_lst)):
                for ix in rng.integers(0, 10000, resamplings):
                    save_path = output_dir / f"{task.benchmark}-{model_name.rsplit('/')[-1]}-{ix:04d}.pkl"
                    s3_path = f"{s3_base_dir}/{save_path.name}"
                    if check_and_download_from_s3(s3_path) and not force:
                        logging.info("Skipping resampling %d for task %s; already exists in S3: %s",
                                     ix, task.benchmark, s3_path)
                        continue

                    # Reset and run
                    logging.info("Running task: model=%s benchmark=%s resampling=%d -> %s",
                                 model_name, task.benchmark, ix, s3_path)
                    activation_storage.clear()

                    # Set random seed for corruption sampling based on resampling index
                    eval_request._alpha = float(rng.uniform(high=corruption_alpha_max))
                    eval_request._corruption_rng = np.random.RandomState(int(ix))
                    try:
                        results[task.benchmark] = run_evaluation(eval_request=eval_request, task_names=[task.benchmark],
                                                                 num_fewshot=task.num_fewshot, batch_size=1, device="auto",
                                                                 limit=limit, seed=int(ix))

                        save_data = {
                            'model': model_name,
                            'task': task.benchmark,
                            'target_layer': target_layer,
                            'layer_activations': activation_storage,
                            'eval_results': results[task.benchmark],
                            'resampling_idx': ix
                        }
                        with open(save_path, 'wb') as f:
                            pickle.dump(save_data, f)
                        upload_to_s3(save_path, s3_base_dir)
                    except:
                        logging.error("Couldn't finish eval=%s", task, exc_info=True)

            if hasattr(eval_request, 'cleanup'):
                eval_request.cleanup()

    finally:
        upload_to_s3(log_path, s3_base_dir)

    return results


@app.function(
    image=MODAL_IMAGE,
    gpu=MODAL_GPU,
    timeout=3600 * 8,
    volumes={
        "/root/outputs": modal.Volume.from_name("parcontrol-data", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("aws"),
    ]
)
def modal_run_instrumented_evaluation(**kwargs):
    """Modal remote function for running instrumented evaluation."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    results = run_instrumented_evaluation(**kwargs)
    logging.info("Modal evaluation completed successfully")
    return results


@app.function(
    image=MODAL_IMAGE,
    gpu=MODAL_GPU,
    timeout=3600 * 24,
    volumes={
        "/root/outputs": modal.Volume.from_name("parcontrol-data", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("aws"),
    ]
)
def modal_do_paired_evaluation(**kwargs):
    """Modal remote function for paired evaluation."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    assert 'corruption_mode' in kwargs, "kwargs missing corruption_mode"
    assert 's3_base_dir' in kwargs, "kwargs missing s3_base_dir"

    logging.info("Modal paired evaluation starting: corruption_mode=%s", kwargs['corruption_mode'])
    results = do_paired_evaluation(**kwargs)
    logging.info("Modal paired evaluation completed successfully")
    return results


@app.function(
    image=MODAL_IMAGE,
    gpu=MODAL_GPU,
    timeout=3600 * 24,
    volumes={
        "/root/outputs": modal.Volume.from_name("parcontrol-data", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("aws"),
    ]
)
def modal_do_dose_response_evaluation(**kwargs):
    """Modal remote function for dose-response evaluation."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    assert 'corruption_mode' in kwargs, "kwargs missing corruption_mode"
    assert 'dose_steps' in kwargs, "kwargs missing dose_steps"

    logging.info("Modal dose-response evaluation starting: corruption_mode=%s, dose_steps=%d",
                 kwargs['corruption_mode'], kwargs['dose_steps'])
    results = do_dose_response_evaluation(**kwargs)
    logging.info("Modal dose-response evaluation completed successfully")
    return results


SUFFIX_MAP = {
    "none": "n",
    "alpha": "c",
    "token": "ct",
    "stream": "cs"
}


def parse_args(argv=None):
    """Parse command line arguments."""
    if argv is None:
        argv = sys.argv[1:]

    now = pd.Timestamp.now(tz="US/Pacific")
    parser = argparse.ArgumentParser(description="PyTorch hook-based activation recording during evaluation")
    parser.add_argument("--model-names", nargs="+", help="Model name or S3 path",
                        default=[
                            "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP-2025-10-07-03-12-51-trial-001",
                            "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-18-44-51",
                            "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-20-48",
                            "s3://obviouslywrong-parcontrol/ParControl/2025-09-24-16-57-26",
                            "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-11-24-40",
                        ])
    parser.add_argument("--target-layer", type=str, default="model.model.norm",
                        help="Target layer for monitoring and optional corruption")
    parser.add_argument("--tasks", nargs="+", default=["truthfulqa_mc2"], choices=[
        "halueval_dialogue",
        "halueval_qa",
        "halueval_summarization",  # HaluEval from Table 1
        "memo-trap_v2",            # MemoTrap from Table 1
        "truthfulqa_mc2",          # TruthfulQA from Table 1
        "nq8",                     # NQ (Natural Questions) from Table 1
        "wikitext",                # Wikitext BPB from Table 1
        "winogrande",              # WG (Winogrande) from Table 1
    ], help="Evaluation tasks")
    parser.add_argument("--output-dir", type=Path, help="Output directory",
                        default=Path(now.strftime("./outputs/neurodiversity/%Y-%m-%d-%H-%M-%S")))
    parser.add_argument("--resamplings", type=int, default=16,
                        help="Number of bootstrap resamplings to take")
    parser.add_argument("--limit", type=int, default=128, help="Sample limit per task")
    parser.add_argument("--s3-base-dir", type=str, default="s3://obviouslywrong-parcontrol/ParControl/neurodiversity",
                        help="S3 base directory for uploading results")
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even if results exist")
    parser.add_argument("--seed", default=int(pd.Timestamp.now().timestamp() % 1000), type=int,
                        help="Which seed to set RNGs to")
    parser.add_argument("--use-modal", action="store_true",
                        help="Run evaluation on Modal with GPU")
    parser.add_argument("--corruption-mode", type=str, default="none", choices=["none", "alpha", "token", "stream"],
                        help="Corruption method: none (disabled), alpha (blending), token (per-token substitution), stream (per-sample fractional)")
    parser.add_argument("--corruption-alpha-max", type=float, default=2.75e-3, help="Maximum alpha for alpha-blending corruption")
    parser.add_argument("--substitute-k-streams", type=int, default=1,
                        help="Number of victim streams for token/stream corruption modes (must be < P)")
    parser.add_argument("--substitute-fraction", type=float, default=0.250,
                        help="Fraction of hidden states to substitute in stream mode (0.0 to 1.0)")
    parser.add_argument("--meta-mode", type=str, default="none", choices=["none", "paired", "dose"],
                        help="Meta-analysis mode: none (single run), paired (vanilla+corrupted), dose (sweep corruption parameter)")
    parser.add_argument("--dose-steps", type=int, default=5,
                        help="Number of sweep points for dose-response mode (including 0)")

    args = parser.parse_args(argv)

    base_dir = args.s3_base_dir.rstrip('/')
    if args.meta_mode == "none":
        suffix = SUFFIX_MAP[args.corruption_mode]
        args.s3_base_dir = f"{base_dir}/evals-N{args.limit}{suffix}"
    elif args.meta_mode == "paired":
        args.s3_base_dir = f"{base_dir}/evals-N{args.limit}p-S{args.seed:03d}"
    elif args.meta_mode == "dose":
        args.s3_base_dir = f"{base_dir}/evals-N{args.limit}d{args.dose_steps}-S{args.seed:03d}"

    return args


def do_paired_evaluation(**kwargs) -> Dict:
    """Run paired vanilla and corrupted evaluations with consistent seed."""
    corruption_mode = kwargs['corruption_mode']
    assert corruption_mode != "none", "Cannot use corruption_mode='none' with paired evaluation"
    base_s3_dir = kwargs['s3_base_dir']

    # things that make runs incompatible with each other within this directory...
    metadata = {
        'corruption_mode': corruption_mode,
        'corruption_alpha_max': kwargs['corruption_alpha_max'],
        'substitute_fraction': kwargs['substitute_fraction'],
        'substitute_k_streams': kwargs['substitute_k_streams'],
    }
    metadata_digest = hashlib.sha256(str(sorted(metadata.items())).encode()).hexdigest()[:6]

    # then add the rest after calculating hexdigest
    now = pd.Timestamp.now(tz="US/Pacific").strftime("%Y-%m-%d-%H-%M-%S")
    metadata |= {
        'limit': kwargs['limit'],
        'models': kwargs['model_names'],
        'resamplings': kwargs['resamplings'],
        'seed': kwargs['seed'],
        'tasks': kwargs['tasks'],
        'timestamp': now,
    }
    metadata_path = Path(kwargs['output_dir']) / f"metadata.{now}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    upload_to_s3(metadata_path, base_s3_dir)
    logging.info("Saved paired metadata: %s", metadata_path)

    # Run corrupted evaluation
    logging.info("Starting paired analysis corrupted run (mode=%s)", corruption_mode)
    corrupted_kwargs = kwargs.copy()
    corrupted_kwargs['s3_base_dir'] = f"{base_s3_dir}/{SUFFIX_MAP[corruption_mode]}-{metadata_digest}"
    upload_to_s3(metadata_path, corrupted_kwargs['s3_base_dir'])
    corrupted_results = run_instrumented_evaluation(**corrupted_kwargs)
    assert corrupted_results is not None, "Corrupted evaluation failed"
    logging.info("Completed paired analysis corrupted run")

    # Run vanilla evaluation
    logging.info("Starting paired analysis vanilla run")
    vanilla_kwargs = kwargs.copy()
    vanilla_kwargs['corruption_mode'] = "none"
    vanilla_kwargs['s3_base_dir'] = f"{base_s3_dir}/{SUFFIX_MAP['none']}"
    upload_to_s3(metadata_path, vanilla_kwargs['s3_base_dir'])
    vanilla_results = run_instrumented_evaluation(**vanilla_kwargs)
    assert vanilla_results is not None, "Vanilla evaluation failed"
    logging.info("Completed paired analysis vanilla run")

    return {
        'vanilla': vanilla_results,
        'corrupted': corrupted_results,
        'metadata': metadata
    }


def do_dose_response_evaluation(dose_steps=5, **kwargs) -> Dict:
    """Run multiple evaluations sweeping corruption parameter for dose-response analysis."""
    corruption_mode = kwargs['corruption_mode']
    assert corruption_mode != "none", "Cannot use corruption_mode='none' with dose-response mode"
    base_s3_dir = kwargs['s3_base_dir']

    # Metadata with digest for compatibility tracking
    metadata = {
        'corruption_mode': corruption_mode,
        'corruption_alpha_max': kwargs['corruption_alpha_max'],
        'substitute_fraction': kwargs['substitute_fraction'],
        'substitute_k_streams': kwargs['substitute_k_streams'],
        'dose_steps': dose_steps,
    }

    if corruption_mode == "alpha":
        sweep_values = np.linspace(0, 1, dose_steps)
        param_name = "corruption_alpha_max"
    elif corruption_mode == "token":
        sweep_values = np.linspace(0.0, 1, dose_steps)
        param_name = "substitute_fraction"
    elif corruption_mode == "stream":
        P = 4  # Note: Fixed P value for stream corruption experiments
        sweep_values = [int(x) for x in np.linspace(0, P-2, min(dose_steps, P-1), dtype=np.int32)]  # int() for JSON serialization
        param_name = "substitute_k_streams"
    else:
        raise ValueError(f"Dose-response not supported for corruption_mode: {corruption_mode}")
    metadata["sweep_param"] = param_name
    metadata["sweep_values"] = list(sweep_values) + [P-1]  # Note: Added P-1 to maintain hash consistency
    metadata_digest = hashlib.sha256(str(sorted(metadata.items())).encode()).hexdigest()[:6]

    logging.warning("Sweeping %s=%s; ignoring specified inputs", param_name, sweep_values)
    # Add remaining metadata
    now = pd.Timestamp.now(tz="US/Pacific").strftime("%Y-%m-%d-%H-%M-%S")
    metadata |= {
        'limit': kwargs['limit'],
        'models': kwargs['model_names'],
        'resamplings': kwargs['resamplings'],
        'seed': kwargs['seed'],
        'tasks': kwargs['tasks'],
        'timestamp': now,
    }

    # NOTE: sample for parallelizability; gotchas: (1) do NOT sample BEFORE hexdigest, (2) MUST enumerate BEFORE sampling.
    dose_results = {}
    for ix, sweep_val in random.sample(list(enumerate(sweep_values)), len(sweep_values)):
        logging.info("Sweeping dose-response %d/%d: %s=%.4f", ix+1, len(sweep_values), param_name, sweep_val)

        # Set corruption parameter for this sweep
        sweep_kwargs = kwargs.copy()
        sweep_kwargs[param_name] = sweep_val
        sweep_kwargs['s3_base_dir'] = f"{base_s3_dir}/{corruption_mode}-{metadata_digest}/{ix:02d}"

        metadata |= {k: v for k, v in sweep_kwargs.items() if k not in ["output_dir"]}
        metadata_path = Path(kwargs['output_dir']) / f"metadata.{now}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        upload_to_s3(metadata_path, sweep_kwargs['s3_base_dir'])

        results = run_instrumented_evaluation(**sweep_kwargs)
        assert results is not None, f"Dose-response evaluation failed at {param_name}={sweep_val}"
        dose_results[sweep_val] = results

    return {
        'dose_response': dose_results,
        'metadata': metadata,
        'param_name': param_name,
        'sweep_values': sweep_values,
    }


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )

    # Dispatch table mapping (meta_mode, use_modal) to functions
    kwargs = {k: v for k, v in vars(args).items() if k not in ['use_modal', 'meta_mode', "dose_steps"]}
    dispatch_table = {
        ("paired", True): lambda: modal_do_paired_evaluation.remote(**kwargs),
        ("paired", False): lambda: do_paired_evaluation(**kwargs),
        ("dose", True): lambda: modal_do_dose_response_evaluation.remote(dose_steps=args.dose_steps, **kwargs),
        ("dose", False): lambda: do_dose_response_evaluation(dose_steps=args.dose_steps, **kwargs),
        ("none", True): lambda: modal_run_instrumented_evaluation.remote(**kwargs),
        ("none", False): lambda: run_instrumented_evaluation(**kwargs),
    }

    handler = dispatch_table.get((args.meta_mode, args.use_modal))
    assert handler is not None, f"Invalid meta_mode: {args.meta_mode}"

    if args.use_modal:
        with modal.enable_output():
            with app.run(detach=True):
                return handler()
    else:
        return handler()


if __name__ == "__main__":
    main()
