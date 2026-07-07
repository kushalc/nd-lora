#!/usr/bin/env python3
"""
Shared statistical testing utilities for Table 1 and Table 2 analyses.
Provides bootstrap confidence intervals, McNemar's test, and sample loading functions.
"""

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from utils.model_checkpoints import S3_BUCKET

# S3 and local cache configuration.
# Raw sample-level eval results live under evals/evals-full/ParControl/P=k/<clean-model-name>/
# in the ndlora bucket; sync_s3_to_local pulls them on demand (no pre-baked parquet).
S3_BASE_PATH = f'{S3_BUCKET}/evals/evals-full/ParControl'
# Shared with build_scores' sync target so table1/table2 reuse the same local copy.
LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / 'leaderboard/evals-full/ParControl'

# Task name to metric key mappings
METRICS_BY_TASK = {
    'halueval_dialogue': 'acc',
    'halueval_qa': 'acc',
    'halueval_summarization': 'acc',
    'memo-trap_v2': 'acc',
    'truthfulqa_mc1': 'acc',
    'truthfulqa_mc2': 'acc',
    'nq8': 'exact_match',
    'nq-swap': 'exact_match',
    'popqa': 'exact_match',
    'triviaqa': 'exact_match',
    'winogrande': 'acc',
    'wikitext': 'bits_per_byte',
}

# Binary classification tasks (use McNemar's test), others use bootstrap
# HaluEval (all variants): binary (hallucinated yes/no)
# MemoTrap: binary (2 choices per question)
# TruthfulQA MC1/MC2: continuous (normalized probability mass on true answers)
# Winogrande: binary (fill-in-blank with 2 options)
# NQ variants, PopQA, TriviaQA: continuous (exact match can have partial credit)
# Wikitext: continuous (bits per byte)
BINARY_TASKS = {'halueval_dialogue', 'halueval_qa', 'halueval_summarization', 'memo-trap_v2', 'winogrande'}


def sync_s3_to_local(force: bool = False) -> None:
    """Sync S3 evals-full directory to local cache.

    If the cache is already populated and force=False, skip the sync — over a slow link an
    incremental `aws s3 sync` is expensive (and would re-download reconstructed aggregate-only
    files whose local size differs from S3). Use force=True to refresh.
    """
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and any(LOCAL_CACHE_DIR.rglob('results_*.json')):
        logging.info("Using populated local cache (skip sync): %s", LOCAL_CACHE_DIR)
        return
    logging.info("Syncing S3 to local cache: %s -> %s", S3_BASE_PATH, LOCAL_CACHE_DIR)
    subprocess.run(['aws', 's3', 'sync', S3_BASE_PATH, str(LOCAL_CACHE_DIR),
                    '--exclude', '*all_results_*'], check=True)
    logging.info("S3 sync complete")


def find_results_files(p_value: str, model_id: str) -> list:
    """Find all results JSON files for a model in local cache."""
    model_dir = LOCAL_CACHE_DIR / p_value / model_id
    assert model_dir.exists(), f"Model directory not found: {model_dir}"

    files = sorted([f for f in model_dir.glob('results_*.json') if 'all_results' not in f.name])
    assert len(files) > 0, f"No results files found in {model_dir}"

    logging.info("Found n_files=%d results files for p_value=%s model_id=%s", len(files), p_value, model_id)
    return files


def load_samples(p_value: str, model_id: str, task_name: str, return_doc_ids: bool = False) -> np.ndarray | tuple | None:
    """
    Load sample-level evaluation results from local cached results files.

    Args:
        p_value: P value directory (e.g., 'P=2')
        model_id: Model timestamp ID
        task_name: Task name
        return_doc_ids: If True, return (values, doc_ids) tuple

    Returns:
        np.ndarray of metric values if return_doc_ids=False
        (values, doc_ids) tuple if return_doc_ids=True
        None if not found
    """
    try:
        result_files = find_results_files(p_value, model_id)
    except AssertionError as e:
        logging.warning("No results files found for p_value=%s model_id=%s task=%s: %s", p_value, model_id, task_name, e)
        return None

    metric_key = METRICS_BY_TASK[task_name]

    # Search through results files for matching task
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Tasks are nested inside data['samples']
        samples_dict = data.get('samples', {})
        available_tasks = list(samples_dict.keys())
        logging.debug("File %s contains tasks: %s", result_file.name, available_tasks)

        # Check if this file contains our task
        if task_name not in samples_dict:
            continue

        # Get samples list directly
        task_samples = samples_dict[task_name]
        assert isinstance(task_samples, list), f"Expected list of samples, got {type(task_samples)}"

        if len(task_samples) == 0:
            logging.warning("File %s has task=%s but empty samples list", result_file.name, task_name)
            continue

        # Extract metric values and doc_ids from samples
        values = []
        doc_ids = []
        for s in task_samples:
            if metric_key in s:
                values.append(s[metric_key])
                if return_doc_ids:
                    doc_ids.append(s['doc_id'])

        if len(values) == 0:
            # Check what keys are actually in samples for debugging
            sample_keys = set()
            for s in task_samples[:5]:
                sample_keys.update(s.keys())
            logging.warning("File %s has task=%s samples but no metric=%s. Sample keys: %s",
                            result_file.name, task_name, metric_key, sample_keys)
            continue

        # Found valid data
        logging.info("Loaded n_samples=%d for model_id=%s task=%s metric=%s from file=%s",
                     len(values), model_id, task_name, metric_key, result_file.name)

        if return_doc_ids:
            return np.array(values), np.array(doc_ids)
        else:
            return np.array(values)

    # Warn if not found
    logging.warning("No valid samples found for task=%s metric=%s in any results file for model_id=%s",
                    task_name, metric_key, model_id)
    return None


def bootstrap_mean_and_se(values: np.ndarray, n_bootstrap: int = 10000, random_state: int = 42) -> tuple:
    """Compute bootstrap mean and standard error."""
    rng = np.random.RandomState(random_state)
    bootstrap_means = np.array([np.mean(rng.choice(values, size=len(values), replace=True))
                                for _ in range(n_bootstrap)])
    return np.mean(values), np.std(bootstrap_means)


def bootstrap_two_sample_test(values_a: np.ndarray, values_b: np.ndarray,
                               n_bootstrap: int = 10000, random_state: int = 42) -> tuple:
    """Perform two-sample bootstrap test. H0: mean(a) - mean(b) <= 0. Returns (p_value, observed_diff)."""
    rng = np.random.RandomState(random_state)
    observed_diff = np.mean(values_a) - np.mean(values_b)

    bootstrap_diffs = np.array([
        np.mean(rng.choice(values_a, size=len(values_a), replace=True)) -
        np.mean(rng.choice(values_b, size=len(values_b), replace=True))
        for _ in range(n_bootstrap)
    ])

    p_value = np.mean(bootstrap_diffs <= 0)
    return p_value, observed_diff


def mcnemar_test(values_a: np.ndarray, values_b: np.ndarray) -> tuple:
    """
    Perform McNemar's test for paired binary data.

    H0: The two models have the same error rate (marginal probabilities equal)
    Returns (chi2_statistic, p_value, contingency_table)
    """
    assert len(values_a) == len(values_b), f"Paired data must have same length: {len(values_a)} vs {len(values_b)}"

    # Build 2x2 contingency table
    # [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    both_correct = np.sum((values_a == 1) & (values_b == 1))
    a_correct_b_wrong = np.sum((values_a == 1) & (values_b == 0))
    a_wrong_b_correct = np.sum((values_a == 0) & (values_b == 1))
    both_wrong = np.sum((values_a == 0) & (values_b == 0))

    contingency = np.array([[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]])

    # McNemar's test focuses on discordant pairs: b and c
    b = a_correct_b_wrong
    c = a_wrong_b_correct
    n_discordant = b + c

    # Handle edge case: no discordant pairs (perfect agreement)
    if n_discordant == 0:
        logging.warning("No discordant pairs - models agree on all samples")
        return np.nan, 1.0, contingency

    # Use exact binomial test if few discordant pairs, otherwise chi-square
    if n_discordant < 100:
        # Exact test: under H0, b ~ Binomial(b+c, 0.5)
        p_value = stats.binomtest(b, n_discordant, 0.5, alternative='greater').pvalue
        chi2 = np.nan
        logging.debug("Using exact binomial McNemar test (b+c=%d < 100)", n_discordant)
    else:
        # Chi-square approximation with continuity correction
        chi2 = (abs(b - c) - 1)**2 / n_discordant
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        logging.debug("Using chi-square McNemar test (b+c=%d >= 25)", n_discordant)

    return chi2, p_value, contingency


def align_paired_samples(values_a: np.ndarray, ids_a: np.ndarray,
                         values_b: np.ndarray, ids_b: np.ndarray) -> tuple:
    """Verify and align paired samples by doc_id. Returns aligned (values_a, values_b)."""
    assert len(ids_a) == len(values_a) and len(ids_b) == len(values_b), "IDs and values length mismatch"

    # Check if already aligned
    if np.array_equal(ids_a, ids_b):
        logging.debug("Samples already aligned (doc_ids match in order)")
        return values_a, values_b

    # Build index mappings
    id_to_val_a = {doc_id: val for doc_id, val in zip(ids_a, values_a)}
    id_to_val_b = {doc_id: val for doc_id, val in zip(ids_b, values_b)}

    # Find common doc_ids
    common_ids = sorted(set(ids_a) & set(ids_b))
    assert len(common_ids) > 0, "No common doc_ids between samples"

    aligned_a = np.array([id_to_val_a[doc_id] for doc_id in common_ids])
    aligned_b = np.array([id_to_val_b[doc_id] for doc_id in common_ids])

    logging.info("Aligned samples: n_a=%d n_b=%d n_common=%d", len(ids_a), len(ids_b), len(common_ids))
    return aligned_a, aligned_b
