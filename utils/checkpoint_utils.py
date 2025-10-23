import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import torch
import torch.nn as nn
import yaml
from botocore.exceptions import ClientError, NoCredentialsError


def construct_s3_key(file_path: Path, output_dir_path: Path, s3_key_prefix: str) -> str:
    """Helper to construct S3 key preserving directory structure relative to output_dir."""
    if output_dir_path and file_path.is_relative_to(output_dir_path):
        # File is inside output directory - preserve relative structure
        rel_path = file_path.relative_to(output_dir_path.parent)  # Include output dir name
        return f"{s3_key_prefix}/{rel_path}".replace('\\', '/') if s3_key_prefix else str(rel_path).replace('\\', '/')
    else:
        # File is outside output directory - use just filename
        return f"{s3_key_prefix}/{file_path.name}" if s3_key_prefix else file_path.name


def upload_to_s3(local_path: str, s3_base_path: str, output_dir: str = None, logger=logging) -> None:
    """Upload a file or directory to S3 using AWS CLI."""
    assert s3_base_path and s3_base_path.startswith('s3://'), \
        f"Invalid S3 path: '{s3_base_path}'. Must start with 's3://'"

    local_path = Path(local_path)
    assert local_path.exists(), f"Local path does not exist: {local_path}"

    # Construct S3 destination path
    s3_base_path = s3_base_path.rstrip('/')

    if output_dir and local_path.is_relative_to(Path(output_dir)):
        # Preserve relative structure from output_dir
        rel_path = local_path.relative_to(Path(output_dir).parent)
        s3_dest = f"{s3_base_path}/{rel_path}"
    else:
        s3_dest = f"{s3_base_path}/{local_path.name}"

    if local_path.is_file():
        # For files, include filename in destination
        cmd = ["aws", "s3", "cp", str(local_path), s3_dest]
    else:
        # For directories, use recursive copy
        cmd = ["aws", "s3", "cp", str(local_path), s3_dest, "--recursive"]

    # Execute AWS CLI command
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"AWS CLI upload failed: {result.stderr}"

    if logger:
        logger.info(f"Uploaded {local_path} to {s3_dest}")


def sync_logs_to_s3(log_dir: str, s3_base_path: str, output_dir: str, logger) -> None:
    """Sync logs directory to S3. Throws exceptions on failure."""
    if not s3_base_path or not s3_base_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 base path: '{s3_base_path}'. Must start with 's3://'")

    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    if not log_path.is_dir():
        raise ValueError(f"Log path is not a directory: {log_dir}")

    upload_to_s3(str(log_path), s3_base_path, output_dir, logger)


def upload_checkpoint_to_s3(checkpoint_path: str, s3_base_path: str, output_dir: str, logger) -> None:
    """Upload checkpoint file to S3. Throws exceptions on failure."""
    if not s3_base_path or not s3_base_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 base path: '{s3_base_path}'. Must start with 's3://'")

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    if not checkpoint_file.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")

    upload_to_s3(str(checkpoint_file), s3_base_path, output_dir, logger)


def save_last_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict[str, Any],
    token_tracker,
    P: int,
    logger,
    basename: str = "last_checkpoint.pt",
    addl_data: dict = {},
) -> str:
    """Save the last (most recent) checkpoint."""
    checkpoint_dir = Path(config["output_dir"]) / "checkpoints" / f"P_{P}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    checkpoint_path = checkpoint_dir / basename
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'processed_tokens': token_tracker.processed_tokens,
        'config': config,
        'P': P
    } | addl_data

    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved last checkpoint at step {step}: {checkpoint_path}")

    # Upload to S3 if requested
    if config.get("sync_to_s3", False):
        upload_checkpoint_to_s3(str(checkpoint_path), config["s3_base_dir"], config["output_dir"], logger)
        upload_to_s3(config_path, config["s3_base_dir"], config["output_dir"], logger)

    return str(checkpoint_path)


def check_and_download_from_s3(s3_key: str, local_path: str = None) -> bool:
    """
    Check if a file exists in S3 and download it if found.

    Args:
        s3_base_path: Base S3 path (e.g., 's3://bucket')
        s3_key: The key to check in S3
        local_path: Where to save the file locally

    Returns:
        bool: True if file was found and downloaded, False otherwise
    """
    if not s3_key or not s3_key.startswith('s3://'):
        raise ValueError(f"Invalid S3 key: '{s3_key}'. Must start with 's3://'")

    # Parse S3 path
    path_parts = s3_key[len("s3://"):].rstrip('/').split('/', 1)
    if len(path_parts) < 2:
        raise ValueError(f"Invalid S3 path format: '{s3_key}'. Expected 's3://bucket/key'")

    bucket_name = path_parts[0]
    s3_client = boto3.client('s3')

    try:
        logging.info("Checking to see if %s exists in S3", s3_key)
        s3_client.head_object(Bucket=bucket_name, Key=path_parts[1])

        # Object exists, download it
        if local_path:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket_name, path_parts[1], local_path)

        logging.info("Found %s in S3", s3_key)
        return True

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404':
            return False
        raise

    except NoCredentialsError as e:
        raise RuntimeError(f"AWS credentials not found: {e}") from e


def load_checkpoint(
    model: nn.Module,
    P: int,
    config: Dict[str, Any],
    logger,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    token_tracker=None,
    checkpoint_path: str = None,
    basename: str = "last_checkpoint.pt",
    require_load: bool = False,
) -> Optional[int]:
    """Load the last checkpoint if it exists. Returns the step to resume from or None."""
    if checkpoint_path is None:
        checkpoint_path = Path(config["output_dir"]) / "checkpoints" / f"P_{P}" / basename
    s3_key = construct_s3_key(checkpoint_path, Path(config["output_dir"]), config["s3_base_dir"])
    if not check_and_download_from_s3(s3_key, str(checkpoint_path)):
        logger.debug("Couldn't find last checkpoint in S3: %s", s3_key)
        return None

    try:
        logger.info("Loading checkpoint from S3: %s", s3_key)
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])

        # Note: Configuration validation for checkpoint loading
        if checkpoint.get('P') != P:
            logger.error("Couldn't properly load last checkpoint; starting from scratch: "
                         f"mismatched checkpoint P={checkpoint.get('P')} != current P={P}")
            return None

        # Load state
        mk, uk = model.load_state_dict(checkpoint['model_state_dict'])
        if mk or uk:
            logger.warning("Couldn't cleanly load model: %d missing keys, %d unexpected keys: %s, %s",
                           len(mk), len(uk), str(mk)[:256] + "..." if len(str(mk)) > 256 else "",
                           str(uk)[:256] + "..." if len(str(uk)) > 256 else "")

        processed_tokens = float("nan")
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if token_tracker:
            token_tracker.processed_tokens = processed_tokens = checkpoint['processed_tokens']
        step = checkpoint['step']

        logger.info(f"Successfully resumed from step {step}: {processed_tokens:,} tokens")
        return step

    except Exception as e:
        logger.error("Couldn't load checkpoint; starting from scratch", exc_info=True)
        if require_load:
            raise
        else:
            return None


def load_best_checkpoint(*nargs, **kwargs):
    return load_checkpoint(*nargs, basename="best_model_state.pt", **kwargs)


def load_last_checkpoint(*nargs, **kwargs):
    return load_checkpoint(*nargs, basename="last_checkpoint.pt", **kwargs)
