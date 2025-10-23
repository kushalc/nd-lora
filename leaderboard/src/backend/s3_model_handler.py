"""
S3 Model Handler for loading ParControl models directly from S3.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
import torch
import yaml

from utils.checkpoint_utils import load_last_checkpoint
from utils.model_utils import build_parscale_model, get_best_available_device, setup_gpu_training

logger = logging.getLogger(__name__)


class S3ModelHandler:
    """Handles downloading and loading ParControl models from S3."""

    def __init__(self, device=None):
        self.s3_client = boto3.client('s3')
        self.cache_dir = Path("outputs")
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device

    def load_model(self, model_path: str, device=None):
        """Load ParControl model directly from S3 or local path.

        Args:
            model_path: S3 path to the model
            device: Device to load model on. If None, uses instance device or auto-detects.
        """

        logger.info("Downloading config.yaml from S3 path %s", model_path)
        config_cache_path = self._get_s3_cache_path(model_path) / "config.yaml"
        config_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._download_s3_file(f"{model_path}/config.yaml", str(config_cache_path))

        with open(config_cache_path, 'r') as f:
            config = yaml.safe_load(f)
            config["s3_base_dir"] = model_path.rsplit("/", 1)[0]  # Note: Extract base directory from model path

        # Determine device to use
        if device is None:
            device = self.device
        if device is None:
            device = get_best_available_device()

        logger.info("Loading model on device: %s", device)
        model, tokenizer = build_parscale_model(config, device=device)

        # Add device to config for checkpoint loading
        config["device"] = device

        # Note: Using last checkpoint for consistency across experiments
        load_last_checkpoint(model=model, P=config["P"], config=config, logger=logger, require_load=True)
        return model, tokenizer

    def _get_s3_cache_path(self, s3_path: str) -> Path:
        """Extract run_id from S3 path for cache directory."""
        path_parts = s3_path.rstrip('/').split('/')
        run_id = path_parts[-1] if path_parts else "unknown"
        return self.cache_dir / run_id

    def _download_s3_file(self, s3_file_path: str, local_file_path: str) -> None:
        """Download a single file from S3."""
        # Parse S3 path
        path_parts = s3_file_path[5:].split('/', 1)
        assert len(path_parts) == 2, f"Invalid S3 path: {s3_file_path}"

        bucket, key = path_parts[0], path_parts[1]

        self.s3_client.download_file(bucket, key, local_file_path)
        logger.info("Downloaded %s to %s", s3_file_path, local_file_path)

    def cleanup(self):
        """Cleanup method for compatibility."""
        pass
