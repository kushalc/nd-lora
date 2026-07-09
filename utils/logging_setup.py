"""
Python logging setup for ParScale experiments.
Provides structured logging with timestamps and JSON-compatible format.
"""

import logging
import os
from datetime import datetime
from typing import Optional

# Get module logger
logger = logging.getLogger(__name__)


def setup_python_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured Python logging for ParScale experiments.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to store log files
        experiment_name: Optional experiment name for log file naming

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_filename = f"parscale_{experiment_name}_{timestamp}.log"
    else:
        log_filename = f"parscale_{timestamp}.log"

    log_filepath = os.path.join(log_dir, log_filename)

    # Get the root logger and clear any existing configuration
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure basic logging with the specified format
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=getattr(logging, level.upper()),
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

    # Get the parscale logger
    logger = logging.getLogger("parscale")
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent propagation to avoid duplication since we're using basicConfig
    logger.propagate = True

    # Log setup completion
    logger.info(f"Logging initialized - Level: {level}, File: {log_filepath}")

    return logger


def log_run_header(logger: logging.Logger, config: dict, git_commit: Optional[str] = None):
    """
    Log experiment run header with configuration and system information.

    Args:
        logger: Logger instance
        config: Full experiment configuration dictionary
        git_commit: Git commit hash for reproducibility
    """
    import platform
    import sys

    import torch

    logger.info("=" * 80)
    logger.info("ParScale Experiment Starting")
    logger.info("=" * 80)

    # System information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

    if git_commit:
        logger.info(f"Git commit: {git_commit}")

    # Configuration
    logger.info("Experiment configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 80)


def log_data_lineage(
    logger: logging.Logger,
    shard_ids: list,
    tokenizer_info: dict,
    data_seed: int
):
    """
    Log data provenance information for reproducibility.

    Args:
        logger: Logger instance
        shard_ids: List of Pile shard IDs used
        tokenizer_info: Tokenizer metadata (name, vocab_size, etc.)
        data_seed: Random seed used for data sampling
    """
    logger.info("Data lineage information:")
    logger.info(f"  Data seed: {data_seed}")
    logger.info(f"  Pile shards: {shard_ids}")
    logger.info(f"  Tokenizer: {tokenizer_info.get('name', 'unknown')}")
    logger.info(f"  Vocab size: {tokenizer_info.get('vocab_size', 'unknown')}")
    logger.info(f"  Special tokens: {tokenizer_info.get('special_tokens', {})}")


METRICS = {
    "loss": "{:6.3f}",
    "perplexity": "{:.3e}",
    "lr": "{:.3e}",
    "progress_pct": "{:6.1%}",
    "processed_tokens": "{:14,d}",
    "tokens_per_second": "{:6.0f}",
}


def log_metrics(
    logger: logging.Logger,
    step: int,
    metrics: dict,
    log_interval: int = 50,
    name: str = "training_step",
):
    """
    Log training step metrics at specified intervals.

    Args:
        logger: Logger instance
        step: Current training step
        metrics: Dictionary of metrics to log
        log_interval: Log every N steps
    """
    if step % log_interval != 0:
        return

    if "processed_tokens" in metrics and "total_time" in metrics:
        metrics["tokens_per_second"] = metrics["processed_tokens"] / metrics["total_time"]
    message = f"{name:15s}={step:6d}"
    for mt, fmt in METRICS.items():
        if mt not in metrics:
            continue
        message += f" | {mt}=" + fmt.format(metrics[mt])
    logger.info(message)


log_training_step = log_metrics
