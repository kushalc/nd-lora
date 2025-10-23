"""
Weights & Biases setup and utilities for ParScale experiments.
Handles initialization, run naming, and artifact management.
"""

import logging
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import torch

import wandb


def setup_wandb(
    config: Dict[str, Any],
    P: int,
    tokens_M: float,
    seq_len: int,
    seed: int,
    project: str = "ParControl",
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
        "python_version": f"{torch.version.__version__}",
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
    logger=None,
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
    logger=None,
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


def save_stream_diagnostics(
    stream_weights: torch.Tensor,
    step: int,
    sample_size: int = 100
):
    """
    Save stream weight diagnostics as W&B artifacts.

    Args:
        stream_weights: Tensor of shape [batch, seq_len, P]
        step: Current training step
        sample_size: Number of samples to save
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy and sample
    weights_np = stream_weights.detach().cpu().numpy()
    if weights_np.shape[0] > sample_size:
        indices = np.random.choice(weights_np.shape[0], sample_size, replace=False)
        weights_np = weights_np[indices]

    # Create heatmap
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Stream usage over sequence positions
    avg_weights = weights_np.mean(axis=0)  # [seq_len, P]
    im1 = axes[0, 0].imshow(avg_weights.T, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Average Stream Weights Over Sequence')
    axes[0, 0].set_xlabel('Sequence Position')
    axes[0, 0].set_ylabel('Stream ID')
    plt.colorbar(im1, ax=axes[0, 0])

    # Stream entropy distribution
    entropy = -np.sum(weights_np * np.log(weights_np + 1e-8), axis=-1)
    axes[0, 1].hist(entropy.flatten(), bins=50, alpha=0.7)
    axes[0, 1].set_title('Stream Weight Entropy Distribution')
    axes[0, 1].set_xlabel('Entropy')
    axes[0, 1].set_ylabel('Frequency')

    # Argmax stream selection
    argmax_streams = np.argmax(weights_np, axis=-1)
    unique, counts = np.unique(argmax_streams, return_counts=True)
    axes[1, 0].bar(unique, counts)
    axes[1, 0].set_title('Stream Selection Frequency (Argmax)')
    axes[1, 0].set_xlabel('Stream ID')
    axes[1, 0].set_ylabel('Selection Count')

    # Mean weight per stream
    mean_weights = weights_np.mean(axis=(0, 1))
    axes[1, 1].bar(range(len(mean_weights)), mean_weights)
    axes[1, 1].set_title('Mean Weight Per Stream')
    axes[1, 1].set_xlabel('Stream ID')
    axes[1, 1].set_ylabel('Mean Weight')

    plt.tight_layout()

    # Save and log to W&B
    filename = f"stream_diagnostics_step_{step}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    wandb.log({f"stream_diagnostics": wandb.Image(filename)}, step=step)

    # Clean up
    plt.close()
    if os.path.exists(filename):
        os.remove(filename)


def log_stream_behavior_metrics(
    stream_disagreement: torch.Tensor,
    is_corrupted: torch.Tensor,
    step: int,
    corruption_types: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """
    Log stream behavior analysis metrics for contrastive learning.

    Args:
        stream_disagreement: Disagreement scores [batch, seq_len] 
        is_corrupted: Boolean corruption indicators [batch]
        step: Current training step
        corruption_types: List of corruption types per example

    Returns:
        Dictionary of stream behavior metrics
    """
    import numpy as np

    # Convert to numpy
    disagreement_np = stream_disagreement.detach().cpu().numpy()
    is_corrupted_np = is_corrupted.detach().cpu().numpy()

    # Compute sequence-level disagreement (mean over tokens)
    seq_disagreement = np.mean(disagreement_np, axis=1)  # [batch]

    # Separate factual vs corrupted
    factual_mask = ~is_corrupted_np
    corrupted_mask = is_corrupted_np

    metrics = {}

    # Agreement metrics
    if factual_mask.any():
        factual_disagreement = seq_disagreement[factual_mask]
        metrics.update({
            "agreement_factual_mean": float(np.mean(factual_disagreement)),
            "agreement_factual_std": float(np.std(factual_disagreement)),
            "agreement_factual_p50": float(np.percentile(factual_disagreement, 50)),
            "agreement_factual_p95": float(np.percentile(factual_disagreement, 95))
        })

    if corrupted_mask.any():
        corrupted_disagreement = seq_disagreement[corrupted_mask]
        metrics.update({
            "agreement_corrupted_mean": float(np.mean(corrupted_disagreement)),
            "agreement_corrupted_std": float(np.std(corrupted_disagreement)),
            "agreement_corrupted_p50": float(np.percentile(corrupted_disagreement, 50)),
            "agreement_corrupted_p95": float(np.percentile(corrupted_disagreement, 95))
        })

        # Separation metric: how well disagreement separates factual vs corrupted
        if factual_mask.any():
            separation = np.mean(corrupted_disagreement) - np.mean(factual_disagreement)
            metrics["separation_score"] = float(separation)

    # Overall metrics
    metrics.update({
        "disagreement_overall_mean": float(np.mean(seq_disagreement)),
        "disagreement_overall_std": float(np.std(seq_disagreement)),
        "batch_corruption_rate": float(np.mean(is_corrupted_np))
    })

    # Corruption type breakdown
    if corruption_types and corrupted_mask.any():
        type_disagreement = {}
        corrupted_disagreement = seq_disagreement[corrupted_mask]
        corrupted_types = [corruption_types[i] for i in range(len(corruption_types)) if is_corrupted_np[i]]

        for i, types in enumerate(corrupted_types):
            for corruption_type in types:
                if corruption_type not in type_disagreement:
                    type_disagreement[corruption_type] = []
                type_disagreement[corruption_type].append(corrupted_disagreement[i])

        for ctype, scores in type_disagreement.items():
            metrics[f"disagreement_{ctype}_mean"] = float(np.mean(scores))

    # Log disagreement histogram
    if len(seq_disagreement) > 0:
        # Create histogram data for W&B
        hist_data = []
        labels = []

        if factual_mask.any():
            hist_data.append(seq_disagreement[factual_mask])
            labels.append("Factual")

        if corrupted_mask.any():
            hist_data.append(seq_disagreement[corrupted_mask])
            labels.append("Corrupted")

        if hist_data:
            try:
                # Log histogram to W&B
                wandb.log({
                    "stream_behavior/disagreement_histogram": wandb.Histogram(
                        np_histogram=np.histogram(seq_disagreement, bins=50)
                    )
                }, step=step)

                # Log separate histograms by corruption status
                for data, label in zip(hist_data, labels):
                    wandb.log({
                        f"stream_behavior/disagreement_hist_{label.lower()}": wandb.Histogram(
                            np_histogram=np.histogram(data, bins=30)
                        )
                    }, step=step)
            except Exception as e:
                logging.warning(f"Failed to log disagreement histograms: {e}")

    return metrics


def log_calibration_data(
    disagreement_scores: torch.Tensor,
    accuracy_scores: torch.Tensor,
    is_corrupted: torch.Tensor,
    step: int
):
    """
    Log calibration data for post-training ROC analysis.

    Args:
        disagreement_scores: Stream disagreement [batch]
        accuracy_scores: Prediction accuracy [batch] 
        is_corrupted: Corruption indicators [batch]
        step: Training step
    """
    try:
        # Convert to lists for JSON serialization
        disagreement_list = disagreement_scores.detach().cpu().tolist()
        accuracy_list = accuracy_scores.detach().cpu().tolist()
        corruption_list = is_corrupted.detach().cpu().tolist()

        # Log as table for later analysis
        calibration_table = wandb.Table(
            columns=["step", "disagreement", "accuracy", "is_corrupted"],
            data=[[step, d, a, c] for d, a, c in zip(disagreement_list, accuracy_list, corruption_list)]
        )

        wandb.log({"calibration_data": calibration_table}, step=step)

    except Exception as e:
        logging.warning(f"Failed to log calibration data: {e}")


def create_dashboard_config() -> Dict[str, Any]:
    """
    Create W&B dashboard configuration for ParScale experiments.

    Returns:
        Dashboard configuration dictionary
    """
    return {
        "charts": [
            {
                "name": "Training Loss by P",
                "type": "line",
                "config": {
                    "x": "step",
                    "y": ["loss/train"],
                    "groupby": ["config.P"]
                }
            },
            {
                "name": "Validation Loss vs Log(P)",
                "type": "scatter",
                "config": {
                    "x": "config.P",
                    "y": ["val/loss"],
                    "transform": "log"
                }
            },
            {
                "name": "Stream Entropy",
                "type": "line",
                "config": {
                    "x": "step",
                    "y": ["stream/entropy_mean"],
                    "groupby": ["config.P"]
                }
            },
            {
                "name": "Stream Agreement by Content Type",
                "type": "line",
                "config": {
                    "x": "step",
                    "y": ["contrastive/agreement_factual_mean", "contrastive/agreement_corrupted_mean"],
                    "groupby": ["config.contrastive.gamma"]
                }
            },
            {
                "name": "Separation Score",
                "type": "line",
                "config": {
                    "x": "step",
                    "y": ["contrastive/separation_score"],
                    "groupby": ["config.P"]
                }
            }
        ]
    }
