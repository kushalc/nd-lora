#!/usr/bin/env python3
"""
Paired statistical analysis for dose-response experiments.

Exploits exact pairing of samples (same resampling_idx → same doc_ids) to maximize
statistical power using per-question paired differences.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_paired_ttest(treatment_df: pd.DataFrame, baseline_df: pd.DataFrame,
                         group_cols: List[str], sample_col: str = 'eval_score') -> pd.DataFrame:
    """
    Compute paired t-tests comparing treatment vs baseline using per-sample differences.

    Args:
        treatment_df: DataFrame with treatment data (one row per sample)
        baseline_df: DataFrame with baseline data (one row per sample)
        group_cols: Columns defining sub-experiments (e.g., ['run_id', 'task', 'resampling_idx'])
        sample_col: Column containing per-sample scores to compare

    Returns:
        DataFrame with one row per sub-experiment containing t-statistics and p-values
    """
    # Reset index if group_cols are in index
    treatment_df = treatment_df.reset_index() if any(col in treatment_df.index.names for col in group_cols) else treatment_df
    baseline_df = baseline_df.reset_index() if any(col in baseline_df.index.names for col in group_cols) else baseline_df

    assert set(group_cols).issubset(treatment_df.columns), \
        f"Missing group_cols in treatment_df: {set(group_cols) - set(treatment_df.columns)}"
    assert set(group_cols).issubset(baseline_df.columns), \
        f"Missing group_cols in baseline_df: {set(group_cols) - set(baseline_df.columns)}"
    assert sample_col in treatment_df.columns, f"Missing {sample_col} in treatment_df"
    assert sample_col in baseline_df.columns, f"Missing {sample_col} in baseline_df"

    # Merge treatment and baseline on group_cols to create paired observations
    merged = treatment_df.merge(baseline_df, on=group_cols, how='inner', suffixes=('_treatment', '_baseline'))

    assert len(merged) > 0, "No paired observations found after merge"

    # Group by sub-experiment and compute paired t-test
    results = []
    for group_key, group_df in merged.groupby(group_cols):
        treatment_scores = group_df[f'{sample_col}_treatment'].values
        baseline_scores = group_df[f'{sample_col}_baseline'].values

        assert len(treatment_scores) == len(baseline_scores), \
            f"Unequal sample sizes for {group_key}: {len(treatment_scores)} vs {len(baseline_scores)}"
        assert len(treatment_scores) >= 1, f"Need >1 paired observations for {group_key}, got {len(treatment_scores)}"

        # Compute paired differences
        diffs = treatment_scores - baseline_scores

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(treatment_scores, baseline_scores)

        # Effect size (Cohen's d for paired samples)
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan

        # Build result dict
        result = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        result.update({
            'n_paired': len(diffs),
            'baseline_mean': baseline_scores.mean(),
            'treatment_mean': treatment_scores.mean(),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
        })
        results.append(result)

    return pd.DataFrame(results)


def meta_analyze_pvalues(pvalues: np.ndarray, method: str = 'fisher') -> Dict[str, float]:
    """
    Combine p-values across sub-experiments using meta-analysis.

    Args:
        pvalues: Array of p-values from independent tests
        method: 'fisher' (Fisher's combined probability) or 'stouffer' (Stouffer's Z-score)

    Returns:
        Dictionary with combined test statistic and combined p-value
    """
    assert len(pvalues) > 0, "Need at least one p-value for meta-analysis"
    assert np.all((pvalues >= 0) & (pvalues <= 1)), f"Invalid p-values (must be in [0,1]): {pvalues}"

    if method == 'fisher':
        # Fisher's combined probability test: -2 * sum(log(p_i)) ~ chi2(2k)
        combined_stat, combined_p = stats.combine_pvalues(pvalues, method='fisher')
        return {
            'method': 'fisher',
            'combined_statistic': combined_stat,
            'combined_p': combined_p,
            'n_tests': len(pvalues),
        }

    elif method == 'stouffer':
        # Stouffer's Z-score method: sum(Z_i) / sqrt(k) ~ N(0,1)
        # Convert two-tailed p-values to Z-scores
        z_scores = stats.norm.ppf(1 - pvalues / 2)
        combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
        combined_p = 2 * (1 - stats.norm.cdf(abs(combined_z)))

        return {
            'method': 'stouffer',
            'combined_statistic': combined_z,
            'combined_p': combined_p,
            'n_tests': len(pvalues),
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'fisher' or 'stouffer'")


def paired_dose_response_analysis(dose_df: pd.DataFrame, baseline_dose: int = 0,
                                  group_cols: Optional[List[str]] = None,
                                  sample_col: str = 'eval_score',
                                  dspec_col: Optional[str] = None,
                                  stratify_cols: Optional[List[str]] = None,
                                  meta_method: str = 'fisher') -> pd.DataFrame:
    """
    Analyze dose-response data using paired t-tests and meta-analysis.

    Args:
        dose_df: DataFrame with columns: dose_level, <group_cols>, <sample_col>
        baseline_dose: Dose level to use as baseline (typically 0)
        group_cols: Columns defining sub-experiments (defaults to ['run_id', 'task', 'resampling_idx'])
        sample_col: Column containing per-sample scores
        dspec_col: Optional column containing dspec values for correlation analysis
        stratify_cols: Optional list of columns to stratify results by (e.g., ['substitute_fraction', 'task'])
        meta_method: Method for combining p-values ('fisher' or 'stouffer')

    Returns:
        DataFrame with one row per (treatment dose, stratification level) containing aggregated statistics
    """
    if group_cols is None:
        group_cols = ['run_id', 'task', 'resampling_idx']

    assert 'dose_level' in dose_df.columns, "dose_df must have 'dose_level' column"
    assert baseline_dose in dose_df['dose_level'].values, f"Baseline dose {baseline_dose} not found in data"

    # Reset index to ensure all columns accessible
    df_reset = dose_df.reset_index()

    # Extract baseline data
    baseline_df = df_reset[df_reset['dose_level'] == baseline_dose].copy()
    treatment_doses = sorted([d for d in df_reset['dose_level'].unique() if d != baseline_dose])

    assert len(treatment_doses) > 0, f"No treatment doses found (baseline={baseline_dose})"

    # Determine stratification levels
    if stratify_cols is not None:
        assert isinstance(stratify_cols, list), f"stratify_cols must be a list, got {type(stratify_cols)}"
        for col in stratify_cols:
            assert col in df_reset.columns, f"Stratification column {col} not found"
        strata = [tuple(x) for x in df_reset[stratify_cols].drop_duplicates().dropna().values]
    else:
        stratify_cols = []
        strata = [None]

    # Analyze each (treatment dose, stratum) combination vs baseline
    summary_results = []
    for stratum in strata:
        for treatment_dose in treatment_doses:
            # Filter by stratum if stratifying
            if stratum is not None:
                strat_filter_treatment = pd.Series([True] * len(df_reset), index=df_reset.index)
                strat_filter_baseline = pd.Series([True] * len(baseline_df), index=baseline_df.index)
                for col, val in zip(stratify_cols, stratum):
                    strat_filter_treatment &= (df_reset[col] == val)
                    strat_filter_baseline &= (baseline_df[col] == val)

                treatment_df = df_reset[(df_reset['dose_level'] == treatment_dose) & strat_filter_treatment].copy()
                baseline_df_stratum = baseline_df[strat_filter_baseline].copy()
            else:
                treatment_df = df_reset[df_reset['dose_level'] == treatment_dose].copy()
                baseline_df_stratum = baseline_df.copy()

            if len(treatment_df) == 0 or len(baseline_df_stratum) == 0:
                continue

            # Paired t-tests for each sub-experiment
            ttest_results = compute_paired_ttest(treatment_df, baseline_df_stratum, group_cols, sample_col)

            if len(ttest_results) == 0:
                continue

            # Meta-analysis across sub-experiments
            try:
                meta_result = meta_analyze_pvalues(ttest_results['p_value'].values, method=meta_method)

                # Compute dspec changes if dspec_col provided
                dspec_change_mean = None
                if dspec_col is not None and dspec_col in df_reset.columns:
                    # Group by sub-experiment and compute dspec changes
                    baseline_dspec = baseline_df_stratum.groupby(group_cols)[dspec_col].mean()
                    treatment_dspec = treatment_df.groupby(group_cols)[dspec_col].mean()
                    common_groups = baseline_dspec.index.intersection(treatment_dspec.index)

                    if len(common_groups) > 0:
                        dspec_change_mean = (treatment_dspec[common_groups] - baseline_dspec[common_groups]).mean()

                # Aggregate statistics for publication table
                result = {
                    'dose': treatment_dose,
                    'Δacc': ttest_results['mean_diff'].mean(),
                    'SE(Δacc)': ttest_results['mean_diff'].std() / np.sqrt(len(ttest_results)),
                    'd': ttest_results['cohens_d'].median(),
                    'p': meta_result['combined_p'],
                    'Δdspec': dspec_change_mean,
                    'N': len(ttest_results),
                }

                if stratify_cols:
                    for col, val in zip(stratify_cols, stratum):
                        result[col] = val

                summary_results.append(result)
            except:
                logging.warning("Couldn't process %s %s", stratum, treatment_dose, exc_info=True)

    return pd.DataFrame(summary_results)


def extract_per_sample_scores(dose_df: pd.DataFrame, eval_results_col: str = 'eval_results',
                              task_col: str = 'task', n_samples: Optional[int] = None,
                              seed: int = 42, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract per-sample (per-question) scores from eval_results for paired analysis.

    Args:
        dose_df: DataFrame with eval_results column containing nested sample data
        eval_results_col: Column name containing eval_results dict
        task_col: Column name containing task name
        n_samples: Optional number of samples to extract per group (maintains pairing across dose levels)
        seed: Random seed for sample selection (default: 42)
        group_cols: Columns defining pairing groups (defaults to ['task', 'resampling_idx'])
                   Sampling finds doc_ids common across dose_level within each group

    Returns:
        Long-format DataFrame with one row per sample (question)
    """
    if group_cols is None:
        group_cols = ['task', 'resampling_idx']

    # Reset index to access all columns
    df_reset = dose_df.reset_index()

    assert eval_results_col in df_reset.columns, f"Missing {eval_results_col} column"
    assert task_col in df_reset.columns, f"Missing {task_col} column"

    long_data = []
    for idx, row in df_reset.iterrows():
        eval_results = row[eval_results_col]
        task = row[task_col]

        # Extract samples for this task
        assert 'samples' in eval_results, f"eval_results missing 'samples' key for index {idx}"
        assert task in eval_results['samples'], f"Task {task} not in samples for index {idx}"

        samples = eval_results['samples'][task]
        assert len(samples) > 0, f"Empty samples for task {task} at index {idx}"

        # Create one row per sample
        for sample in samples:
            sample_row = {k: v for k, v in row.items() if k != eval_results_col}
            sample_row.update({
                'doc_id': sample['doc_id'],
                'eval_score': sample['acc'],  # Per-question accuracy
            })
            long_data.append(sample_row)

    df_long = pd.DataFrame(long_data)

    # Subsample if requested, maintaining pairing within groups
    if n_samples is not None:
        assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
        for col in group_cols:
            assert col in df_long.columns, f"Group column {col} not found in extracted data"

        rng = np.random.RandomState(seed)
        sampled_data = []

        # Process each pairing group separately
        for group_key, group_df in df_long.groupby(group_cols):
            # Find doc_ids common across all dose levels in this group
            dose_col = 'dose_level' if 'dose_level' in group_df.columns else None
            if dose_col is not None:
                common_doc_ids = group_df.groupby('doc_id')[dose_col].nunique()
                expected_doses = group_df[dose_col].nunique()
                common_doc_ids = common_doc_ids[common_doc_ids == expected_doses].index.tolist()
            else:
                # No dose column, just use all doc_ids
                common_doc_ids = group_df['doc_id'].unique().tolist()

            n_available = len(common_doc_ids)
            n_to_sample = min(n_samples, n_available)

            assert n_to_sample > 0, \
                f"Group {group_key}: no common doc_ids available across dose levels"

            # Sample N doc_ids from this group
            sampled_doc_ids = rng.choice(common_doc_ids, size=n_to_sample, replace=False)
            sampled_data.append(group_df[group_df['doc_id'].isin(sampled_doc_ids)])

        df_long = pd.concat(sampled_data, ignore_index=True)

    return df_long
