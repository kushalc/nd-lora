#!/usr/bin/env python3
"""
Script to analyze ParControl experiment results and generate data for paper tables.
Processes parquet files to extract metrics for Table 1 and Table 2.
"""

import argparse
import logging

import numpy as np
import pandas as pd
# On-demand builder for the aggregated single-stream score parquet (from raw S3 evals).
from build_scores import ensure_scores_parquet
# Import shared statistical testing utilities
from statsig_utils import (BINARY_TASKS, align_paired_samples,
                           bootstrap_two_sample_test, load_samples,
                           mcnemar_test, sync_s3_to_local)

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(funcName)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S'
)

# Task name mappings from display names to file names
TASK_FILE_MAPPINGS = {
    'HaluEval Dialog, Accuracy': 'halueval_dialogue',
    'HaluEval QA, Accuracy': 'halueval_qa',
    'HaluEval Summarization, Accuracy': 'halueval_summarization',
    'MemoTrap, Accuracy': 'memo-trap_v2',
    'TruthfulQA MC1, Accuracy': 'truthfulqa_mc1',
    'TruthfulQA MC2, Accuracy': 'truthfulqa_mc2',
    'NQ (8-shot), EM': 'nq8',
    'nq swap, EM': 'nq-swap',
    'popQA, EM': 'popqa',
    'TriviaQA (8-shot), EM': 'triviaqa',
    'winogrande, Accuracy': 'winogrande',
    'wikitext, bits_per_byte': 'wikitext',
}

# Model ID mappings for different P values and configurations.
# model_id is the eval-dir name; evals are stored under clean model names (see utils/model_checkpoints.py).
MODEL_ID_MAPPINGS = {
    'P=1': {
        'Repro LoRA R32': 'Qwen2.5-0.5B_P1_R32'
    },
    'P=2': {
        'ND-LoRA [OptC9]': 'ND-LoRA_P2',
        'SharedLoRA R32': 'ParScale_P2_R32',
    },
    'P=4': {
        'ND-LoRA [OptC9]': 'ND-LoRA_P4',
        'SharedLoRA R64': 'ParScale_P4_R64',
    },
    'P=8': {
        'ND-LoRA [OptC9]': 'ND-LoRA_P8',
        'SharedLoRA R128': 'ParScale_P8_R128',
    },
}


def compute_significance_for_task(task_col_name: str, best_p: str, baseline_p: str = 'P=1',
                                  test_method: str = 'auto', seed: int = 42) -> dict:
    """
    Compute statistical significance for a single task comparing best P vs baseline.

    Args:
        task_col_name: Column name from parquet (e.g., 'HaluEval Dialog, Accuracy')
        best_p: Best performing P value (e.g., 'P=2')
        baseline_p: Baseline P value (default: 'P=1')
        test_method: 'auto', 'bootstrap', 'mcnemar', or 'both'
        seed: Random seed for bootstrap

    Returns:
        dict with significance test results
    """
    # Skip if best_p equals baseline (no comparison needed)
    if best_p == baseline_p:
        return {'significant': False, 'p_value': 1.0, 'test_used': 'none'}

    # Map task column name to file name
    if task_col_name not in TASK_FILE_MAPPINGS:
        logging.warning("Task %s not found in mappings, skipping significance test", task_col_name)
        return {'significant': False, 'p_value': np.nan, 'test_used': 'unknown'}

    task_file = TASK_FILE_MAPPINGS[task_col_name]
    is_binary = task_file in BINARY_TASKS

    # Determine test method
    if test_method == 'auto':
        use_mcnemar = is_binary
    else:
        use_mcnemar = test_method in ('mcnemar', 'both')

    logging.info("Computing significance for task=%s best_p=%s vs baseline_p=%s (binary=%s)",
                 task_col_name, best_p, baseline_p, is_binary)

    # Load baseline model data
    baseline_model_id = MODEL_ID_MAPPINGS.get(baseline_p, {}).get('Repro LoRA R32')
    if baseline_model_id is None:
        logging.warning("No model ID found for baseline_p=%s", baseline_p)
        return {'significant': False, 'p_value': np.nan, 'test_used': 'no_baseline'}

    # Find best model ID for best_p (try OptC9 variants first)
    best_model_ids = MODEL_ID_MAPPINGS.get(best_p, {})
    best_model_id = best_model_ids.get('ND-LoRA [OptC9]') or best_model_ids.get('SharedLoRA R32')

    if best_model_id is None:
        logging.warning("No model ID found for best_p=%s", best_p)
        return {'significant': False, 'p_value': np.nan, 'test_used': 'no_best_model'}

    # Load samples
    try:
        if use_mcnemar:
            baseline_data = load_samples(baseline_p, baseline_model_id, task_file, return_doc_ids=True)
            best_data = load_samples(best_p, best_model_id, task_file, return_doc_ids=True)

            if baseline_data is None or best_data is None:
                logging.warning("Missing sample data for task=%s", task_col_name)
                return {'significant': False, 'p_value': np.nan, 'test_used': 'missing_data'}

            baseline_samples, baseline_ids = baseline_data
            best_samples, best_ids = best_data
            baseline_aligned, best_aligned = align_paired_samples(baseline_samples, baseline_ids, best_samples, best_ids)

            # Run McNemar test
            chi2, mcnemar_p, contingency = mcnemar_test(best_aligned, baseline_aligned)
            logging.info("McNemar test: chi2=%.4f p=%.4f", chi2 if not np.isnan(chi2) else 0.0, mcnemar_p)

            return {'significant': mcnemar_p < 0.05, 'p_value': mcnemar_p, 'test_used': 'mcnemar', 'chi2': chi2}

        else:  # Bootstrap
            baseline_samples = load_samples(baseline_p, baseline_model_id, task_file, return_doc_ids=False)
            best_samples = load_samples(best_p, best_model_id, task_file, return_doc_ids=False)

            if baseline_samples is None or best_samples is None:
                logging.warning("Missing sample data for task=%s", task_col_name)
                return {'significant': False, 'p_value': np.nan, 'test_used': 'missing_data'}

            # Run bootstrap test
            bootstrap_p, diff = bootstrap_two_sample_test(best_samples, baseline_samples, random_state=seed)
            logging.info("Bootstrap test: diff=%.4f p=%.4f", diff, bootstrap_p)

            return {'significant': bootstrap_p < 0.05, 'p_value': bootstrap_p, 'test_used': 'bootstrap', 'diff': diff}

    except Exception as e:
        logging.error("Error computing significance for task=%s: %s", task_col_name, e, exc_info=True)
        return {'significant': False, 'p_value': np.nan, 'test_used': 'error'}


def analyze_table2(parquet_path=None,
                   compute_stats: bool = False, test_method: str = 'auto', seed: int = 42):
    """Extract Table 2: Task-dependent optimality using Qwen LoRA R32 (P=1) as baseline, OptC9 only.

    NOTE: the parquet MultiIndex model levels (e.g. 'Repro LoRA R32', 'OptC9') are produced by
    analyze_experiments.parse_model_metadata() from the clean eval-dir names. Verify these filter
    strings against a freshly generated parquet after the first on-demand build.
    """
    if parquet_path is None:
        parquet_path = ensure_scores_parquet(plot_type="pub", mode="full")
    df = pd.read_parquet(parquet_path)
    df_q05 = df.loc[df.index.get_level_values(0) == 'Q0.5B'].copy()

    task_cols = [
        ('Hallucination', 'HaluEval (Dialog)', 'HaluEval Dialog, Accuracy'),
        ('Hallucination', 'HaluEval (QA)', 'HaluEval QA, Accuracy'),
        ('Hallucination', 'HaluEval (Summ)', 'HaluEval Summarization, Accuracy'),
        ('Hallucination', 'MemoTrap v2', 'MemoTrap, Accuracy'),
        ('Hallucination', 'TruthfulQA (MC1)', 'TruthfulQA MC1, Accuracy'),
        ('Hallucination', 'TruthfulQA (MC2)', 'TruthfulQA MC2, Accuracy'),
        ('General', 'NQ (8-shot)', 'NQ (8-shot), EM'),
        ('General', 'NQ-swap', 'nq swap, EM'),
        ('General', 'PopQA', 'popQA, EM'),
        ('General', 'TriviaQA (8-shot)', 'TriviaQA (8-shot), EM'),
        ('General', 'Winogrande', 'winogrande, Accuracy'),
        ('General', 'Wikitext', 'wikitext, bits_per_byte'),
    ]

    # Extract baseline: Qwen LoRA R32 (P=1)
    baseline_key = ('Q0.5B', 'P=1', 'Repro LoRA R32', 'ParControl Q0.5B P=1: Repro LoRA R32')
    assert baseline_key in df.index, f"Baseline {baseline_key} not found in dataframe"
    baseline_row = df.loc[baseline_key]

    # Filter to OptC9 treatments plus all P=1 Repro-LoRA ranks: the per-P "best score" for
    # knowledge tasks (P*=1) is the best single-stream baseline across ranks (e.g. R128), while
    # the Delta% is still measured against the R32 baseline row below.
    lora_level = df_q05.index.get_level_values(2)
    df_optc9 = df_q05.loc[lora_level.str.contains("|".join(["OptC9", "Repro LoRA"]), na=False)].copy()

    # Group by P and take max across OptC9 variants for best scores
    p_level = df_optc9.index.get_level_values(1)
    max_by_p = df_optc9.groupby(p_level).max()

    # Build results DataFrame with MultiIndex
    categories, task_names, col_names = zip(*task_cols)

    # Extract baseline scores using vectorized list comprehension
    baseline_scores = [baseline_row[col] for col in col_names]
    assert all(score > 0 for score in baseline_scores), "Invalid baseline values found"

    # Identify metrics where lower is better
    lower_is_better_cols = {'wikitext, bits_per_byte'}

    # Find best P and best score using vectorized operations
    best_scores = [max_by_p[col].min() if col in lower_is_better_cols else max_by_p[col].max() for col in col_names]
    best_p_indices = [
        max_by_p[col].idxmin() if col in lower_is_better_cols else max_by_p[col].idxmax() for col in col_names
    ]

    # Calculate improvements using vectorized operations
    abs_deltas = [best - baseline for best, baseline in zip(best_scores, baseline_scores)]
    rel_deltas = [(delta / baseline) * 100 for delta, baseline in zip(abs_deltas, baseline_scores)]

    # Invert deltas for lower-is-better metrics and zero out P=1 cases
    for i, col in enumerate(col_names):
        if col in lower_is_better_cols:
            abs_deltas[i] = -abs_deltas[i]
            rel_deltas[i] = -rel_deltas[i]
        if best_p_indices[i] == 'P=1':
            abs_deltas[i] = 0
            rel_deltas[i] = 0

    # Create MultiIndex DataFrame
    index = pd.MultiIndex.from_tuples(list(zip(categories, task_names)), names=['Category', 'Task'])
    results_df = pd.DataFrame(
        {
            'Baseline (Qwen R32)': baseline_scores,
            'Best P': [p.replace('P=', '') for p in best_p_indices],
            'Best Score': best_scores,
            'Abs Δ': abs_deltas,
            'Rel Δ%': rel_deltas,
        },
        index=index,
    )

    # Compute statistical significance if requested
    if compute_stats:
        logging.info("Computing statistical significance tests for Table 2...")
        sync_s3_to_local(force=False)

        sig_results = []
        for col_name, best_p in zip(col_names, best_p_indices):
            sig_result = compute_significance_for_task(col_name, best_p, baseline_p='P=1', test_method=test_method, seed=seed)
            sig_results.append(sig_result)

        # Add significance columns to results
        results_df['Significant'] = [r['significant'] for r in sig_results]
        results_df['p-value'] = [r['p_value'] for r in sig_results]
        results_df['Test'] = [r['test_used'] for r in sig_results]

        # Log summary
        n_sig = sum(results_df['Significant'])
        logging.info("Significance summary: n_significant=%d/%d tasks", n_sig, len(results_df))

    logging.info(
        "Table 2 Analysis (Baseline: Qwen LoRA R32):\n%s",
        results_df.to_string(float_format=lambda x: f'{x:.3f}' if abs(x) < 10 else f'{x:.1f}'),
    )
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Table 1 and Table 2 analyses with optional significance tests')
    parser.add_argument('--no-compute-stats', action='store_false', dest="compute_stats",
                        help='Skip the statistical significance tests for Table 2 (which require S3 eval data)')
    parser.add_argument('--test-method', choices=['bootstrap', 'mcnemar', 'both', 'auto'], default='auto',
                        help='Statistical test method (default: auto - McNemar for binary tasks, bootstrap for others)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrap (default: 42)')
    parser.add_argument('--summary-parquet', type=str, default=None,
                        help='Path to aggregated-results parquet (default: build on demand from raw S3 evals)')
    args = parser.parse_args()

    analyze_table2(parquet_path=args.summary_parquet, compute_stats=args.compute_stats,
                   test_method=args.test_method, seed=args.seed)
