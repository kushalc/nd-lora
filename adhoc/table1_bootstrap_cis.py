#!/usr/bin/env python3
"""Compute bootstrap confidence intervals and significance tests for Table 1 using full evaluation results from S3."""

import argparse
import logging

import numpy as np
import pandas as pd

# Import shared statistical testing utilities
from statsig_utils import (
    BINARY_TASKS, align_paired_samples, bootstrap_mean_and_se, bootstrap_two_sample_test, load_samples, mcnemar_test,
    sync_s3_to_local,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(funcName)s %(message)s')

# model_id is the eval-dir name; evals are stored under clean model names (see model_checkpoints_paper.py).
MODEL_MAPPINGS = {
    'ND-LoRA P=2': ('P=2', 'ND-LoRA_P2'),
    'ParScale P=2': ('P=2', 'ParScale_P2_R32'),
    'Qwen R32': ('P=1', 'Qwen2.5-0.5B_P1_R32'),
}

TASK_MAPPINGS = {
    'halueval_summarization': 'HaluEval',
    'memo-trap_v2': 'MemoTrap',
    'truthfulqa_mc2': 'TruthfulQA',
    'nq8': 'NQ',
    'winogrande': 'WG',
}


def main(force_sync: bool = False, test_method: str = 'auto', seed: int = 42):
    """Compute bootstrap statistics for Table 1."""
    logging.info("="*80)
    logging.info("BOOTSTRAP CONFIDENCE INTERVALS FOR TABLE 1")
    logging.info("Using test method: %s", test_method)
    logging.info("Random seed: %d", seed)
    logging.info("="*80)

    # Sync S3 to local cache
    sync_s3_to_local(force=force_sync)

    rows = []
    for task_file, task_display in TASK_MAPPINGS.items():
        logging.info("="*80)
        logging.info("Processing task=%s (%s)", task_display, task_file)
        logging.info("="*80)

        # Determine which test to use based on test_method and task type
        is_binary = task_file in BINARY_TASKS
        if test_method == 'auto':
            use_mcnemar = is_binary
            use_bootstrap = not is_binary
            logging.info("Auto-selected test method: %s (task is %s)",
                        'McNemar' if use_mcnemar else 'Bootstrap',
                        'binary' if is_binary else 'non-binary')
        else:
            use_mcnemar = test_method in ('mcnemar', 'both')
            use_bootstrap = test_method in ('bootstrap', 'both')

        # Load samples with or without doc_ids depending on test method
        if use_mcnemar:
            # Load with doc_ids for McNemar's test
            model_data = {}
            for model_name, (p_value, model_id) in MODEL_MAPPINGS.items():
                result = load_samples(p_value, model_id, task_file, return_doc_ids=True)
                model_data[model_name] = result

            nd_data = model_data['ND-LoRA P=2']
            qwen_data = model_data['Qwen R32']
            par_data = model_data['ParScale P=2']

            if nd_data is None or qwen_data is None:
                logging.warning("SKIPPING task=%s - missing required model data (nd=%s qwen=%s)",
                                task_display, nd_data is not None, qwen_data is not None)
                continue

            nd_samples, nd_ids = nd_data
            qwen_samples, qwen_ids = qwen_data

            # Align paired samples
            nd_aligned, qwen_aligned = align_paired_samples(nd_samples, nd_ids, qwen_samples, qwen_ids)
        else:
            # Load without doc_ids for bootstrap only
            model_samples = {model_name: load_samples(p_value, model_id, task_file)
                             for model_name, (p_value, model_id) in MODEL_MAPPINGS.items()}

            nd_samples = model_samples['ND-LoRA P=2']
            qwen_samples = model_samples['Qwen R32']
            par_samples = model_samples['ParScale P=2']

            if nd_samples is None or qwen_samples is None:
                logging.warning("SKIPPING task=%s - missing required model data (nd=%s qwen=%s)",
                                task_display, nd_samples is not None, qwen_samples is not None)
                continue

            nd_aligned, qwen_aligned = nd_samples, qwen_samples

        # Initialize row
        row = {'task': task_display}

        # Compute bootstrap statistics if requested
        if use_bootstrap:
            nd_mean, nd_se = bootstrap_mean_and_se(nd_aligned, random_state=seed)
            qwen_mean, qwen_se = bootstrap_mean_and_se(qwen_aligned, random_state=seed)
            bootstrap_p, diff = bootstrap_two_sample_test(nd_aligned, qwen_aligned, random_state=seed)

            logging.info("ND-LoRA P=2: mean=%.4f se=%.4f", nd_mean, nd_se)
            logging.info("Qwen R32: mean=%.4f se=%.4f", qwen_mean, qwen_se)
            logging.info("ND-LoRA vs Qwen (Bootstrap): diff=%.4f p=%.4f sig=%s", diff, bootstrap_p, bootstrap_p < 0.05)

            row.update({
                'nd_mean': nd_mean,
                'nd_se': nd_se,
                'qwen_mean': qwen_mean,
                'qwen_se': qwen_se,
                'bootstrap_p': bootstrap_p,
                'bootstrap_sig': bootstrap_p < 0.05,
            })

        # Compute McNemar's test if requested
        if use_mcnemar:
            chi2, mcnemar_p, contingency = mcnemar_test(nd_aligned, qwen_aligned)
            logging.info("ND-LoRA vs Qwen (McNemar): chi2=%.4f p=%.4f sig=%s",
                         chi2 if not np.isnan(chi2) else 0.0, mcnemar_p, mcnemar_p < 0.05)
            logging.info("Contingency table:\n%s", contingency)

            row.update({
                'mcnemar_chi2': chi2,
                'mcnemar_p': mcnemar_p,
                'mcnemar_sig': mcnemar_p < 0.05,
            })

        # Determine overall significance
        if use_bootstrap and use_mcnemar:
            # Both tests must agree for significance
            row['significant'] = row['bootstrap_sig'] and row['mcnemar_sig']
        elif use_bootstrap:
            row['significant'] = row['bootstrap_sig']
        else:  # mcnemar only
            row['significant'] = row['mcnemar_sig']

        # Handle ParScale if available (bootstrap only for now)
        if use_bootstrap:
            if use_mcnemar and par_data is not None:
                par_samples, par_ids = par_data
            elif not use_mcnemar and par_samples is not None:
                pass  # already have par_samples
            else:
                par_samples = None

            if par_samples is not None:
                par_mean, par_se = bootstrap_mean_and_se(par_samples, random_state=seed)
                logging.info("ParScale P=2: mean=%.4f se=%.4f", par_mean, par_se)
                row.update({'par_mean': par_mean, 'par_se': par_se})
            else:
                logging.warning("ParScale P=2 data not available for task=%s", task_display)

        rows.append(row)

    # Warn about completion status
    n_completed = len(rows)
    n_total = len(TASK_MAPPINGS)
    if n_completed < n_total:
        logging.warning("PARTIAL RESULTS: completed=%d/%d tasks - evaluation still running or incomplete",
                        n_completed, n_total)
    else:
        logging.info("SUCCESS: All %d tasks completed successfully", n_total)

    assert n_completed > 0, "No tasks completed - evaluation data not available yet"

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    logging.info("="*80)
    logging.info("RESULTS DATAFRAME")
    logging.info("="*80)
    logging.info("\n%s", df.to_string(index=False))

    # Generate LaTeX table entries
    logging.info("="*80)
    logging.info("LATEX TABLE FORMAT")
    logging.info("="*80)
    for _, row in df.iterrows():
        sig = "*" if row['significant'] else ""

        if use_bootstrap:
            nd_str = f"{row['nd_mean']:.3f}±{row['nd_se']:.3f}{sig}"
            qwen_str = f"{row['qwen_mean']:.3f}±{row['qwen_se']:.3f}"
            par_str = f"{row['par_mean']:.3f}±{row['par_se']:.3f}" if 'par_mean' in row else "N/A"
        else:
            # McNemar only - show mean without SE
            nd_str = f"{row['task']} (McNemar p={row['mcnemar_p']:.4f}){sig}"
            qwen_str = "N/A"
            par_str = "N/A"

        logging.info("\n%s:", row['task'])
        if use_bootstrap:
            logging.info("  ND-LoRA R16 (P=2):  %s", nd_str)
            logging.info("  ParScale R64 (P=2): %s", par_str)
            logging.info("  Qwen R64:           %s", qwen_str)
        if use_mcnemar:
            logging.info("  McNemar chi2=%.4f p=%.4f",
                         row['mcnemar_chi2'] if not np.isnan(row['mcnemar_chi2']) else 0.0,
                         row['mcnemar_p'])
        if use_bootstrap and use_mcnemar:
            logging.info("  Bootstrap p=%.4f", row['bootstrap_p'])

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute bootstrap CIs for Table 1 from S3 full eval results')
    parser.add_argument('--force', action='store_true', help='Force re-sync from S3 even if local cache exists')
    parser.add_argument('--test-method', choices=['bootstrap', 'mcnemar', 'both', 'auto'], default='auto',
                        help='Statistical test method (default: auto - McNemar for binary tasks, bootstrap for others)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrap (default: 42)')
    args = parser.parse_args()
    main(force_sync=args.force, test_method=args.test_method, seed=args.seed)
