#!/usr/bin/env python3
"""
Reusable evaluation utilities for S3 upload, idempotence checks, and task management.
Extracted from eval_experiments.py and backend_cli.py for reuse across evaluation scripts.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from utils.checkpoint_utils import upload_to_s3


def check_s3_task_exists(s3_base_dir: str, model_alias: str, task_name: str) -> Optional[str]:
    """
    Check if evaluation results for a specific task already exist in S3.

    Args:
        s3_base_dir: Base S3 directory (e.g., 's3://bucket/path')
        model_alias: Model name or alias to check
        task_name: Specific task name to check for

    Returns:
        S3 path if task results exist, None otherwise
    """
    assert s3_base_dir and s3_base_dir.startswith('s3://'), \
        f"Invalid S3 path: '{s3_base_dir}'. Must start with 's3://'"

    # Parse S3 path
    path_parts = s3_base_dir.split('/')
    assert len(path_parts) >= 3, f"Invalid S3 path format: '{s3_base_dir}'"

    bucket_name = path_parts[2]
    s3_key_prefix = '/'.join(path_parts[3:]) + f"/{model_alias}/"

    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_key_prefix
        )

        if 'Contents' not in response:
            return None

        for obj in response['Contents']:
            key = obj['Key']
            # Check result files (results_YYYY-MM-DD-HH-MM-SS.json)
            if "results_" not in key or not key.endswith('.json'):
                continue

            try:
                result_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = result_obj['Body'].read().decode('utf-8')
                result_data = json.loads(content)
                s3_path = f"s3://{bucket_name}/{key}"

                # Check if task exists in results
                config = result_data.get('config', {})
                if task_name in config.get('task_name', '') or task_name in str(config):
                    return s3_path
                elif 'results' in result_data and task_name in str(result_data):
                    return s3_path
                elif task_name in result_data and isinstance(result_data[task_name], dict) and 'error' not in result_data[task_name]:
                    return s3_path
            except Exception:
                logging.warning("Couldn't read results: %s", key, exc_info=True)
                continue

        return None

    except ClientError:
        logging.warning("Failed to check S3 for existing results", exc_info=True)
        return None
    except Exception:
        logging.warning("Unexpected error checking S3", exc_info=True)
        return None


def save_evaluation_results(
    results: Dict[str, Any],
    model_alias: str,
    output_base_dir: str,
    s3_base_dir: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Save evaluation results to local filesystem and optionally upload to S3.

    Args:
        results: Dictionary of evaluation results
        model_alias: Model name or alias for directory structure
        output_base_dir: Base directory for output files
        s3_base_dir: Optional S3 base directory for upload
        timestamp: Optional timestamp string (defaults to current time)

    Returns:
        Path to saved results file
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now(tz="US/Pacific").strftime("%Y-%m-%d-%H-%M-%S")

    output_path = os.path.join(output_base_dir, *model_alias.split("/"), f"results_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: '<not serializable>')

    # Upload to S3 if requested
    if s3_base_dir:
        upload_to_s3(output_path, s3_base_dir + f"/{model_alias}", None, logging.getLogger(__name__))

    logging.info("Results saved to: %s", output_path)
    return output_path


def get_model_alias_from_s3_path(model_name: str) -> str:
    """
    Extract model alias from S3 path or return model name as-is.

    Args:
        model_name: Model name or S3 path

    Returns:
        Model alias string
    """
    if not model_name.startswith('s3://'):
        return model_name

    # Try to load config from outputs directory
    config_file = Path("outputs") / model_name.split("/")[-1] / "config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                P = config.get('P', 'unknown')
                run_id = Path(config.get('output_dir')).name
                return f"ParControl/P={P}/{run_id}"
        except Exception:
            logging.warning("Failed to load config for model alias", exc_info=True)

    # Fallback to last component of S3 path
    return model_name.split("/")[-1]


def should_skip_task(
    task_name: str,
    model_alias: str,
    s3_base_dir: Optional[str],
    force: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Check if a task should be skipped based on existing S3 results.

    Args:
        task_name: Name of the task to check
        model_alias: Model name or alias
        s3_base_dir: S3 base directory to check
        force: If True, never skip tasks

    Returns:
        Tuple of (should_skip, s3_path_if_exists)
    """
    if force or not s3_base_dir:
        return False, None

    s3_path = check_s3_task_exists(s3_base_dir, model_alias, task_name)
    if s3_path:
        logging.info("Skipping task %s for model %s; result already exists in S3: %s",
                     task_name, model_alias, s3_path)
        return True, s3_path

    return False, None


def aggregate_results(
    results_dict: Dict[str, Any],
    output_dir: str,
    model_alias: str,
    s3_base_dir: Optional[str] = None
) -> str:
    """
    Aggregate and save all evaluation results.

    Args:
        results_dict: Dictionary of all task results
        output_dir: Base output directory
        model_alias: Model name or alias
        s3_base_dir: Optional S3 base directory for upload

    Returns:
        Path to aggregated results file
    """
    timestamp = pd.Timestamp.now(tz="US/Pacific").strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(output_dir, *model_alias.split("/"), f"all_results_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=lambda o: '<not serializable>')

    # Upload aggregated results to S3
    if s3_base_dir:
        upload_to_s3(output_path, s3_base_dir + f"/{model_alias}", None, logging.getLogger(__name__))

    logging.info("All results saved to: %s", output_path)
    return output_path


def batch_check_s3_tasks(
    s3_base_dir: str,
    model_alias: str,
    task_names: List[str]
) -> Dict[str, Optional[str]]:
    """
    Check multiple tasks for existing S3 results in a single operation.

    Args:
        s3_base_dir: Base S3 directory
        model_alias: Model name or alias
        task_names: List of task names to check

    Returns:
        Dictionary mapping task name to S3 path if exists
    """
    results = {}
    for task_name in task_names:
        results[task_name] = check_s3_task_exists(s3_base_dir, model_alias, task_name)
    return results