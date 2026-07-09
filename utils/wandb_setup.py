"""
Weights & Biases setup and utilities for ND-LoRA experiments.
Handles initialization, run naming, and artifact management.
"""

import logging
import os
import platform
import subprocess
from typing import Any, Dict, Optional

import psutil
import torch

import wandb

from utils.model_checkpoints import WANDB_PROJECT


def setup_wandb(
    config: Dict[str, Any],
    P: int,
    tokens_M: float,
    seq_len: int,
    seed: int,
    project: str = WANDB_PROJECT,
    group: str = "qwen25-0.5b_ctp",
    job_type: str = "train",
    offline_mode: bool = False,
    run_id: Optional[str] = None
) -> wandb.run:
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        config: Full experiment configuration dictionary
        P: Number of parallel streams
        tokens_M: Target tokens in millions
        seq_len: Sequence length
        seed: Random seed
        project: W&B project name
        group: W&B group for organizing runs
        job_type: W&B job type
        offline_mode: Whether to run in offline mode
        run_id: Optional W&B run ID for resuming runs

    Returns:
        W&B run object
    """
    # Set offline mode if requested or if WANDB_MODE is set
    if offline_mode or os.environ.get("WANDB_MODE") == "offline":
        os.environ["WANDB_MODE"] = "offline"

    # Get git commit for reproducibility
    git_commit = config.get("git_commit") or get_git_commit()

    # Build tags list
    tags = [f"P={P}", f"tokens={tokens_M:.0f}M", f"seq_len={seq_len}"]
    if git_commit:
        tags.append(f"commit={git_commit[:8]}")  # Add short commit hash as tag

    # Initialize W&B run with optional run_id for resuming
    run = wandb.init(
        project=project,
        group=group,
        job_type=job_type,
        config=config,
        tags=tags,
        id=run_id,
        resume="allow" if run_id else None
    )

    # Log additional metadata
    wandb.config.update({
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "system_info": get_system_info()
    }, allow_val_change=True)

    return run


def get_git_commit() -> Optional[str]:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_system_info() -> Dict[str, Any]:
    """Get system information for logging."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_count = psutil.cpu_count()

        return {
            "total_memory_gb": round(memory.total / (1024**3), 2),
            "cpu_count": cpu_count,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
    except Exception:
        return {}


def log_training_metrics(
    step: int,
    metrics: Dict[str, Any],
    loss_components: Optional[Dict[str, Any]] = None,
    stream_stats: Optional[Dict[str, Any]] = None,
    system_stats: Optional[Dict[str, Any]] = None,
    contrastive_stats: Optional[Dict[str, Any]] = None,
    commit: bool = True,
):
    """
    Log training metrics to W&B.

    Args:
        step: Current training step
        metrics: Training metrics dictionary
        loss_components: Loss component breakdown (optional)
        stream_stats: Stream diagnostics (optional)
        system_stats: System resource metrics (optional)
        contrastive_stats: Contrastive learning stream behavior metrics (optional)
        commit: Whether to commit this log entry (set False if more logs coming for same step)
    """
    log_dict = {}

    # Add training metrics
    for key, value in metrics.items():
        if key.startswith(('loss', 'lr', 'grad_norm', 'tokens', 'step_time', "processed_tokens")):
            log_dict[key] = value

    # Add loss components breakdown (NEW)
    if loss_components:
        for key, value in loss_components.items():
            log_dict[f"loss_components/{key}"] = value

    # Add stream diagnostics
    if stream_stats:
        for key, value in stream_stats.items():
            log_dict[f"stream/{key}"] = value

    # Add contrastive learning stream behavior metrics
    if contrastive_stats:
        for key, value in contrastive_stats.items():
            log_dict[f"contrastive/{key}"] = value

    # Add system metrics
    if system_stats:
        for key, value in system_stats.items():
            log_dict[f"system/{key}"] = value

    wandb.log(log_dict, step=step, commit=commit)


def log_validation_metrics(
    step: int,
    val_metrics: Dict[str, Any],
):
    """
    Log validation metrics to W&B.

    Args:
        step: Current training step
        val_metrics: Validation metrics dictionary
    """
    log_dict = {}
    for key, value in val_metrics.items():
        log_dict[f"val/{key}"] = value

    wandb.log(log_dict, step=step, commit=True)


def monitor_system_resources() -> Dict[str, float]:
    """
    Monitor system resources and return metrics.

    Returns:
        Dictionary of system metrics
    """
    try:
        # Memory usage
        memory = psutil.virtual_memory()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Disk usage
        disk = psutil.disk_usage('/')

        # MPS memory if available
        mps_allocated = 0
        if torch.backends.mps.is_available():
            try:
                mps_allocated = torch.mps.current_allocated_memory() / (1024**2)  # MB
            except Exception:
                pass

        return {
            "mem_used_gb": round((memory.total - memory.available) / (1024**3), 2),
            "mem_percent": memory.percent,
            "cpu_percent": cpu_percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "mps_allocated_mb": round(mps_allocated, 2)
        }
    except Exception:
        return {}
