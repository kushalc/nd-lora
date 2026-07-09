"""
Analysis utilities for ParScale experiments.
Handles results analysis, plotting, and final report generation.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# NOTE: leaderboard.src.{envs,utils} are imported lazily inside parse_leaderboard_results'
# download branch — they pull in HF-leaderboard internals that aren't needed (and don't import)
# for the paper-repro path (--no-download / S3-only).

# Get module logger
logger = logging.getLogger(__name__)


def parse_leaderboard_results(
    results_path: str,
    model_whitelist: Optional[List[str]] = None,
    eval_blacklist: Optional[List[Tuple[Optional[str], Optional[str]]]] = None,
    model_name_mapping: Optional[Dict[str, str]] = None,
    download_repos: bool = True,
    s3_path: Optional[str] = None,
    min_ct: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Parse leaderboard evaluation results from JSON files and generate structured data.

    Args:
        results_path: Path to directory containing evaluation result JSON files
        model_whitelist: List of model name patterns to include (None = include all)
        eval_blacklist: List of (metric_regex, dataset_regex) tuples to exclude
        model_name_mapping: Dictionary mapping model names to display names
        download_repos: Whether to download from HuggingFace repos (requires imports)
        s3_path: Optional S3 path for syncing ParControl model results (e.g., f"{S3_BUCKET}/evals/quick")

    Returns:
        Tuple of:
            - DataFrame with parsed results in wide format (models as columns, metrics as rows)
            - Raw data dictionary mapping model_name -> {(dataset, metric): value}
    """
    import os
    import subprocess

    # Default values
    if eval_blacklist is None:
        eval_blacklist = [
            ("stderr", None),
            ("f1", None),
            (None, "faithdial"),
            (None, "truthfulqa_gen"),
            (None, "fever"),
        ]

    if model_whitelist is None:
        model_whitelist = [
            "ParControl/",
            "Qwen/Qwen2.5-0.5B",
        ]

    if model_name_mapping is None:
        model_name_mapping = {}

    # Download repos if requested
    if download_repos:
        from leaderboard.src.envs import RESULTS_REPO
        from leaderboard.src.utils import my_snapshot_download
        logger.info("Syncing results from HF: %s to %s", s3_path, results_path)
        my_snapshot_download(repo_id=RESULTS_REPO, revision="main",
                             local_dir=results_path, repo_type="dataset", max_workers=60)

        logger.info("Syncing ParControl results from S3: %s to %s", s3_path, results_path)
        try:
            os.makedirs(results_path, exist_ok=True)
            cmd = ["aws", "s3", "sync", s3_path, results_path]
            # Run without capturing output so progress is shown on screen
            result = subprocess.run(cmd, check=True)
            logger.info("Successfully synced ParControl models from S3")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to sync from S3 with exit code %s", e.returncode, exc_info=True)
            raise
        except FileNotFoundError:
            logger.error("AWS CLI not found. Please install aws-cli to use S3 sync functionality")
            raise

    # Find all JSON files
    result_path_lst = []
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file.endswith(".json"):
                result_path_lst.append(os.path.join(root, file))

    # Parse results
    model_dataset_metric_to_result_map = {}
    data_map = {}

    for path in result_path_lst:
        # Check model whitelist
        if not any(name in path for name in model_whitelist):
            continue

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            model_name = data.get("config", {}).get("model_name")
            if not model_name:
                logger.debug("Skipped %s without model_name", path)
                continue

            # ParControl S3 evals store a legacy run-id path in config.model_name; the eval-dir
            # was renamed to the clean model name during the bucket migration. Key off the clean
            # dir name (…/P=k/<clean>/results_*.json) so the clean MODEL_NAMES mapping applies.
            # HF leaderboard results (config.model_name = "org/model") are left untouched.
            if isinstance(model_name, str) and model_name.startswith("ParControl/"):
                model_name = os.path.basename(os.path.dirname(path))

            for dataset_name, results_dict in data["results"].items():
                for metric_name, value in results_dict.items():
                    to_add = True

                    # Apply blacklist
                    for metric_regex, data_regex in eval_blacklist:
                        if metric_regex is not None and metric_regex in metric_name:
                            to_add = False
                            break
                        elif data_regex is not None and data_regex in dataset_name:
                            to_add = False
                            break

                    # Special filtering rules
                    if 'bertscore' in metric_name:
                        if 'precision' not in metric_name:
                            to_add = False

                    if 'halueval' in dataset_name:
                        if 'acc' not in metric_name:
                            to_add = False

                    if 'ifeval' in dataset_name:
                        if 'prompt_level_strict_acc' not in metric_name:
                            to_add = False

                    if 'squad' in dataset_name:
                        if 'best_exact' in metric_name:
                            to_add = False

                    if "truthfulqa_gen" in dataset_name:
                        if "acc" not in metric_name:
                            to_add = False
                        if "rouge" in metric_name:
                            to_add = False

                    if ('xsum' in dataset_name or 'cnn' in dataset_name) and 'v2' not in dataset_name:
                        to_add = False

                    # Check if value is numeric
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            to_add = False

                    logging.info("Saw %s %s -> %s", dataset_name, metric_name, to_add)
                    if to_add:
                        # Apply value transformations
                        if 'rouge' in metric_name:
                            value /= 100.0

                        if 'squad' in dataset_name:
                            value /= 100.0

                        # Sanitize names
                        sanitised_metric_name = metric_name
                        if "," in sanitised_metric_name:
                            sanitised_metric_name = sanitised_metric_name.split(',')[0]
                        sanitised_metric_name = _sanitise_metric(sanitised_metric_name)
                        sanitised_dataset_name = _sanitise_dataset(dataset_name)

                        # Apply model name mapping
                        mapped_model_name = model_name_mapping.get(model_name, model_name)

                        if mapped_model_name not in data_map:
                            data_map[mapped_model_name] = {}

                        subkey = (sanitised_dataset_name, sanitised_metric_name)
                        key = (mapped_model_name,) + subkey

                        # Handle conflicts by taking minimum value
                        if key in model_dataset_metric_to_result_map:
                            old_value = model_dataset_metric_to_result_map[key]
                            if np.abs(old_value - value) > 1e-3:
                                result = min(old_value, value)
                                logger.warning("Chose minimum value for conflicted key=%s: %.3f %.3f -> %.3f",
                                               key, value, old_value, result)
                                model_dataset_metric_to_result_map[key] = result
                                data_map[mapped_model_name][subkey] = result
                        else:
                            model_dataset_metric_to_result_map[key] = value
                            data_map[mapped_model_name][subkey] = value

        except Exception:
            logger.error("Couldn't parse %s", path, exc_info=True)

    # Convert to DataFrame format
    # Restructure data_map for DataFrame: rows = (dataset, metric), columns = models
    data_map_v2 = {}
    for model_name in data_map.keys():
        for dataset_metric in data_map[model_name].keys():
            if dataset_metric not in data_map_v2:
                data_map_v2[dataset_metric] = {}
            data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]

    # Create DataFrame
    df = pd.DataFrame.from_dict(data_map_v2, orient='index')
    df.index = [', '.join(map(str, idx)) for idx in df.index]

    # Filter rows with sufficient data
    counts_s = df.count(axis=1)
    df = df.reindex(counts_s[counts_s >= min_ct].index)

    # Sort lexically
    df = df.sort_index(axis=0, key=lambda x: x.str.lower())
    df = df.reindex(sorted(df.columns), axis=1)

    # Set small values to NaN for better visualization
    df[df < 1e-3] = np.nan

    return df, data_map


def _sanitise_metric(name: str) -> str:
    """Sanitise metric name for display."""
    res = name
    res = res.replace("prompt_level_strict_acc", "Prompt-Level Accuracy")
    res = res.replace("acc", "Accuracy")
    res = res.replace("exact_match", "EM")
    res = res.replace("avg-selfcheckgpt", "AVG")
    res = res.replace("max-selfcheckgpt", "MAX")
    res = res.replace("rouge", "ROUGE-")
    res = res.replace("bertscore_precision", "BERT-P")
    res = res.replace("exact", "EM")
    res = res.replace("HasAns_EM", "HasAns")
    res = res.replace("NoAns_EM", "NoAns")
    res = res.replace("em", "EM")
    return res


def _sanitise_dataset(name: str) -> str:
    """Sanitise dataset name for display."""
    res = name
    res = res.replace("tqa8", "TriviaQA (8-shot)")
    res = res.replace("nq8", "NQ (8-shot)")
    res = res.replace("nq_open", "NQ (64-shot)")
    res = res.replace("triviaqa", "TriviaQA (64-shot)")
    res = res.replace("truthfulqa", "TruthfulQA")
    res = res.replace("ifeval", "IFEval")
    res = res.replace("selfcheckgpt", "SelfCheckGPT")
    res = res.replace("truefalse_cieacf", "True-False")
    res = res.replace("mc", "MC")
    res = res.replace("race", "RACE")
    res = res.replace("squad", "SQuAD")
    res = res.replace("memo-trap", "MemoTrap")
    res = res.replace("cnndm", "CNN/DM")
    res = res.replace("xsum", "XSum")
    res = res.replace("qa", "QA")
    res = res.replace("summarization", "Summarization")
    res = res.replace("dialogue", "Dialog")
    res = res.replace("halueval", "HaluEval")
    res = res.replace("_v2", "")
    res = res.replace("_", " ")
    return res
