"""
Model utilities for ParScale experiments.
Handles model loading, PEFT setup, and ParScale configuration.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)

from ParScale.configuration_qwen2_parscale import Qwen2ParScaleConfig
from ParScale.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM
from utils.lora_ablation import filter_lora_modules_for_ablation
from utils.stream_aware_lora import (StreamAwareLoRA,
                                     replace_linear_with_stream_lora)

# Get module logger
logger = logging.getLogger(__name__)

# Add src directory to path to import ParScale modules
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)


def get_best_available_device() -> str:
    """
    Detect and return the best available device for computation.
    Priority: CUDA > MPS > CPU

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = "cpu"
        logger.info("Using CPU device")

    return device


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base Qwen2.5 model and tokenizer.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on. If None, auto-detects best available device.

    Returns:
        Tuple of (model, tokenizer)
    """
    # Auto-detect device if not specified
    if device is None or device == "auto":
        device = get_best_available_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    dtype = torch.float32
    attn_impl = None
    if str(device).startswith("cuda"):
        pass
    elif str(device).startswith("mps"):
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, config=base_config,
                                                 torch_dtype=dtype, attn_implementation=attn_impl,
                                                 device_map=device if device != "auto" else None,
                                                 trust_remote_code=True)

    return model, tokenizer


def create_parscale_config(
    base_config,
    P: int,
    prefix_len: int = 48,
    attn_smooth: float = 0.1
) -> Qwen2ParScaleConfig:
    """
    Create ParScale configuration from base model config.

    Args:
        base_config: Base model configuration
        P: Number of parallel streams
        prefix_len: Length of prefix tokens per layer per stream
        attn_smooth: Attention smoothing parameter

    Returns:
        ParScale configuration
    """
    # Convert base config to ParScale config
    parscale_config = Qwen2ParScaleConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
        hidden_act=getattr(base_config, 'hidden_act', 'silu'),
        max_position_embeddings=getattr(base_config, 'max_position_embeddings', 32768),
        initializer_range=getattr(base_config, 'initializer_range', 0.02),
        rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
        use_cache=True,
        tie_word_embeddings=getattr(base_config, 'tie_word_embeddings', False),
        rope_theta=getattr(base_config, 'rope_theta', 10000.0),
        rope_scaling=getattr(base_config, 'rope_scaling', None),
        use_sliding_window=getattr(base_config, 'use_sliding_window', False),
        sliding_window=getattr(base_config, 'sliding_window', 4096),
        max_window_layers=getattr(base_config, 'max_window_layers', 28),
        attention_dropout=getattr(base_config, 'attention_dropout', 0.0),
        # ParScale specific parameters
        parscale_n=P,
        parscale_n_tokens=prefix_len,
        parscale_attn_smooth=attn_smooth
    )

    return parscale_config


def convert_to_parscale_model(
    base_model: PreTrainedModel,
    parscale_config: Qwen2ParScaleConfig,
    device: str = "auto"
) -> Qwen2ParScaleForCausalLM:
    """
    Convert base model to ParScale model.

    Args:
        base_model: Base Qwen2 model
        parscale_config: ParScale configuration
        device: Device to place model on

    Returns:
        ParScale model
    """
    # Create new ParScale model
    parscale_model = Qwen2ParScaleForCausalLM(parscale_config)

    # Copy weights from base model (excluding ParScale-specific components)
    base_state = base_model.state_dict()
    parscale_state = parscale_model.state_dict()

    # Copy compatible weights
    for key in parscale_state.keys():
        if key in base_state and parscale_state[key].shape == base_state[key].shape:
            parscale_state[key].copy_(base_state[key])

    # Initialize ParScale-specific parameters
    _initialize_parscale_parameters(parscale_model, parscale_config)

    # Move to device
    if device != "auto":
        parscale_model = parscale_model.to(device)

    return parscale_model


def _initialize_parscale_parameters(
    model: Qwen2ParScaleForCausalLM,
    config: Qwen2ParScaleConfig
):
    """Initialize ParScale-specific parameters."""
    # Initialize prefix parameters
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'prefix_k'):
            nn.init.normal_(layer.self_attn.prefix_k, std=config.initializer_range)
        if hasattr(layer.self_attn, 'prefix_v'):
            nn.init.normal_(layer.self_attn.prefix_v, std=config.initializer_range)

    # Initialize aggregation layer
    if hasattr(model.model, 'aggregate_layer'):
        for module in model.model.aggregate_layer:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def setup_peft_model(
    model: PreTrainedModel,
    P: int,
    training_mode: str = "peft",
    use_stream_lora: bool = False,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_ablation_enabled=False,
    lora_ablation_modules=None,
    lora_ablation_layers=None,
) -> Union[PeftModel, PreTrainedModel]:
    """
    Setup model for PEFT or full training with optional LoRA ablation.

    Args:
        model: Base model to configure
        P: Number of parallel streams
        training_mode: "peft" or "full"
        use_stream_lora: Use stream-aware LoRA instead of standard PEFT
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_modules: List of modules to apply LoRA to
        lora_ablation_enabled: Whether to enable LoRA ablation
        lora_ablation_modules: List of modules to ablate
        lora_ablation_layers: List of layers to ablate

    Returns:
        Configured model (PeftModel for PEFT, original for full)
    """
    if training_mode == "full":
        # For full training, enable all parameters
        for param in model.parameters():
            param.requires_grad = True

        return model

    elif training_mode == "peft":
        # For PEFT, freeze base model and enable only ParScale or LoRA parameters
        # First freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Apply LoRA ablation by filtering target_modules and layers
        effective_lora_modules = lora_modules.copy()
        layers_to_transform = None
        if lora_ablation_enabled:

            # Get number of layers from model config
            total_layers = len(model.model.layers)
            effective_lora_modules, layers_to_transform = filter_lora_modules_for_ablation(
                lora_modules,
                enabled=True,
                ablated_modules=lora_ablation_modules or [],
                ablated_layers=lora_ablation_layers or [],
                total_layers=total_layers
            )
            logger.info(f"LoRA ablation enabled: filtered modules from {lora_modules} to {effective_lora_modules}")

        # Standard LoRA setup (L-1 uses same LoRA, just different loss)
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 target_modules=effective_lora_modules, layers_to_transform=layers_to_transform, bias="none")

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        if use_stream_lora:
            model = replace_linear_with_stream_lora(
                model=model,
                P=P,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )

        if P > 1:
            # Unfreeze prefix & aggregate layer parameters
            for layer in model.base_model.model.model.layers:
                layer.self_attn.prefix_k.requires_grad = True
                layer.self_attn.prefix_v.requires_grad = True

            for param in model.base_model.model.model.aggregate_layer.parameters():
                param.requires_grad = True

        return model

    else:
        raise ValueError(f"Unknown training_mode: {training_mode}")


def build_parscale_model(config, device=None):
    # Use provided device or get from config or auto-detect
    if device is None:
        device = config.get("device", None)
    if device is None or device == "auto":
        device = get_best_available_device()

    base_model, tokenizer = load_base_model(config["model_name"], device=device)

    # Create ParScale configuration
    logger.info(f"Creating ParScale config with P={config['P']}")
    parscale_config = create_parscale_config(base_model.config, P=config["P"],
                                             prefix_len=config["prefix_len"],
                                             attn_smooth=0.1)

    # Convert to ParScale model
    logger.info("Converting to ParScale model...")
    model = convert_to_parscale_model(base_model, parscale_config, device=device)

    # Setup training mode (PEFT or full)
    logger.info(f"Setting up {config['training_mode']} training...")
    lora_modules = config.get("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = setup_peft_model(model, config["P"], config["training_mode"], lora_rank=config["lora_rank"],
                             use_stream_lora=config["use_stream_lora"], lora_modules=lora_modules,
                             lora_ablation_enabled=config.get("lora_ablation_enabled", False),
                             lora_ablation_modules=config.get("lora_ablation_modules", []),
                             lora_ablation_layers=config.get("lora_ablation_layers", []))
    validate_parscale_model(model, config["P"])
    return model, tokenizer


def count_trainable_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Count trainable parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Count ParScale-specific and LoRA parameters
    parscale_params = 0
    lora_params = 0
    stream_lora_params = 0

    # Check for StreamAwareLoRA modules
    has_stream_lora = any(isinstance(m, StreamAwareLoRA) for m in model.modules())

    # Count parameters by name pattern (more reliable)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name:
                if has_stream_lora:
                    stream_lora_params += param.numel()
                else:
                    lora_params += param.numel()
            elif 'lora' in name.lower() or 'adapter' in name.lower():
                lora_params += param.numel()
            elif 'prefix_k' in name or 'prefix_v' in name or 'aggregate_layer' in name:
                parscale_params += param.numel()

    result = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "parscale_parameters": parscale_params,
        "trainable_percentage": (trainable_params / total_params) * 100
    }

    if stream_lora_params > 0:
        result["stream_lora_parameters"] = stream_lora_params
    else:
        result["lora_parameters"] = lora_params

    return result


def setup_gpu_training(high_memory_mode: bool = False,
                       enable_mixed_precision: bool = True
                       ) -> Tuple[torch.device, Optional[torch.cuda.amp.GradScaler]]:
    """
    Setup optimized training backend with high-memory optimizations.

    Args:
        high_memory_mode: Enable high-memory optimizations
        enable_mixed_precision: Enable mixed precision training

    Returns:
        Tuple of (device, grad_scaler)
    """
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Use GradScaler for mixed precision on CUDA
        device = torch.device("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=enable_mixed_precision)

        torch.backends.cudnn.benchmark = True  # Good for consistent shapes
        torch.backends.cuda.matmul.allow_tf32 = True  # Performance boost
        torch.backends.cudnn.allow_tf32 = True  # Performance boost

        torch.cuda.empty_cache()
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(high_memory_mode)  # Can be memory heavy
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(high_memory_mode)  # Can cause OOM
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(high_memory_mode)  # Safe fallback

        # Clean memory state
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        scaler = None

        if high_memory_mode:
            # MPS optimizations for 20GB unified memory systems
            try:
                torch.backends.mps.enable_native_attention()
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.85'  # Use 85% of unified memory
            except AttributeError:
                logger.info("MPS basic optimizations enabled")

            # Enable optimized allocator for large memory
            try:
                torch.mps.empty_cache()  # Clear cache
                torch.mps.set_per_process_memory_fraction(0.9)  # Use 90% of unified memory
            except:
                pass

    else:
        device = torch.device("cpu")
        scaler = None

        if high_memory_mode:
            # CPU optimizations for 20GB+ system RAM
            torch.set_num_threads(min(24, torch.get_num_threads() * 3))  # More aggressive threading
            torch.set_num_interop_threads(min(8, torch.get_num_interop_threads() * 2))

            # Enable MKLDNN optimizations for large memory systems
            try:
                torch.backends.mkldnn.enabled = True
                torch.backends.mkldnn.verbose = 0  # Silent optimizations
                logger.info("CPU 20GB optimizations: increased threading, MKLDNN enabled")
            except:
                logger.info("CPU 20GB optimizations: increased threading enabled")

    logger.info(f"Training device: {device}")
    logger.info(f"Mixed precision: {scaler is not None}")

    return device, scaler


def get_model_memory_usage(model: PreTrainedModel) -> Dict[str, float]:
    """
    Get model memory usage statistics.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with memory usage in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "parameters_mb": param_size / (1024 * 1024),
        "buffers_mb": buffer_size / (1024 * 1024),
        "total_mb": total_size / (1024 * 1024)
    }


def validate_parscale_model(model: Union[PeftModel, PreTrainedModel], P: int):
    if isinstance(model, PeftModel):
        model = model.base_model.model

    if not hasattr(model.model, 'parscale_n'):
        assert P == 1  # Non-ParScale model is valid only for P=1
    else:
        assert model.model.parscale_n == P

    if P > 1:
        # Check prefix parameters
        for layer in model.model.layers:
            assert (hasattr(layer.self_attn, 'prefix_k') and
                    hasattr(layer.self_attn, 'prefix_v'))

            assert (layer.self_attn.prefix_k.shape[0] == P and
                    layer.self_attn.prefix_v.shape[0] == P)

        # Check aggregation layer
        assert hasattr(model.model, 'aggregate_layer')

    logger.info("Validated ParScale model: %s", model)
