#!/usr/bin/env python3
"""
Analyze ParScale experiments and generate leaderboard plots.
Combines data parsing from leaderboard results with plotting for absolute and relative scores.

RUN THIS:
source .env; ./analyze_experiments.py; open plots/pub-full-*.png
"""

import argparse
import logging
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.analysis_utils import parse_leaderboard_results
from utils.model_checkpoints_paper import MODEL_NAMES

BASELINE_RANK = {
    "P=1": "R16",
    "P=2": "R32",
    "P=4": "R64",
    "P=8": "R128",
}


PUB_EVAL_BLACKLIST = {
    "SQuADv2",
    "True-False",
}

PUB_MODEL_BLACKLIST = [
    # r"\bOSL IndLoRA\b",
    "MvO",
    "CAG",
    "RandK3",
]


def parse_model_metadata(name):
    """Extract model metadata from column names."""
    base = pd.NA
    p_value = pd.NA
    treatment = pd.NA

    # Parse base model
    match = re.search(r'(Q\d+(?:\.\d+)?[B])', name)
    if match:
        base = match.group(1)
    elif "Qwen" in name:
        base = name.split("/")[-1].replace("Qwen2.5-", "Q")

    # Parse P value
    match = re.search(r'(P=\d+)', name)
    if match:
        p_value = match.group(1)

    # Parse treatment
    if ": " in name:
        treatment = name.split(": ")[-1]

    return base, p_value, treatment


def prettify_multiindex(df, axis=1):
    """Convert multiindex to readable column names."""
    df = df.copy()
    if axis == 1:
        df.columns = [" | ".join(map(str, x)) for x in df.columns]
    else:
        df.index = [" | ".join(map(str, x)) for x in df.index]
    return df


def get_model_ordering(df_with_metadata):
    """Extract model ordering based on mean performance across evaluations."""
    model_means = df_with_metadata.mean(axis=1)
    model_ordering = model_means.sort_values(ascending=False).index
    return model_ordering


def sort_models_within_clusters(df, model_ordering):
    """Sort models within each (base, P) cluster according to provided ordering."""
    if not isinstance(df.index, pd.MultiIndex):
        return df

    # Group by base and P value, then sort within each group
    sorted_indices = []

    for (base, p_val), group in df.groupby(level=[0, 1]):
        # Get models in this cluster
        cluster_models = group.index

        # Find ordering for models in this cluster
        ordered_models = [idx for idx in model_ordering if idx in cluster_models]

        # Add any models not in ordering (fallback)
        remaining_models = [idx for idx in cluster_models if idx not in ordered_models]
        ordered_models.extend(remaining_models)

        sorted_indices.extend(ordered_models)

    # Reindex dataframe with sorted indices
    return df.reindex(sorted_indices)


def generate_absolute_plots(df, output_dir, plot_type='all', analysis_mode="quick", model_ordering=None):
    """Generate absolute score heatmaps with subplots for base model size and # of streams."""
    output_path = Path(output_dir)

    # Set up plotting parameters
    kwargs = {
        "square": True,
        "annot": True,
        "annot_kws": {"size": 8},
        "cmap": "viridis",
        "vmin": 0,
        "vmax": 1,
        "fmt": ".1%",
        "cbar": False,
    }

    # Transpose and clean
    df = df.sort_index().T
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Get model ordering and sort models within clusters
    if model_ordering is None:
        model_ordering = get_model_ordering(df)
    df = sort_models_within_clusters(df, model_ordering)

    # Get unique base models and P values
    base_models = df.index.get_level_values(0).unique().dropna()
    p_values = df.index.get_level_values(1).unique().dropna()

    if len(base_models) == 0:
        logging.warning("No valid models found for plotting")
        return None

    # Create 2D grid: rows = base models, cols = P values
    n_rows = len(base_models)
    n_cols = len(p_values)

    # Calculate width ratios based on number of columns in each subplot
    widths = []
    for base_model in base_models:
        for p_value in p_values:
            try:
                mask = ((df.index.get_level_values(0) == base_model) &
                        (df.index.get_level_values(1) == p_value))
                subset = df.loc[mask]
                if len(subset) > 0:
                    widths.append(len(subset))
            except (KeyError, IndexError):
                pass

    # Calculate figure size based on total width ratios
    total_width = sum(widths)
    fig_width = max(16, total_width * 0.8)
    fig_height = 12

    fig, axs = plt.subplots(1, len(widths), figsize=(fig_width, fig_height),
                            tight_layout=True, sharey=True,
                            gridspec_kw={'width_ratios': widths})
    axs = axs.reshape(-1)

    for ix, (name, subset_df) in enumerate(df.groupby(level=[0, 1])):
        ax = axs[ix]

        sns.heatmap(subset_df.droplevel([0, 1, -1]).T, ax=ax, **kwargs)
        ax.set_title(f"{name[0]} | {name[1]}")
        ax.tick_params(axis='x', labelsize=10, rotation=90)

        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # Save plot
    plot_file = output_path / f'{plot_type}-{analysis_mode}-zabsolute.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info("Generated absolute scores plot: %s", plot_file)
    return df


def generate_relative_plots(df, output_dir, plot_type='all', analysis_mode="quick", model_ordering=None, baseline_mode='single-stream'):
    """Generate relative score difference heatmaps with subplots for base model size and # of streams.

    Args:
        baseline_mode: 'stream-matched' (default) - compare with same P value baseline
                      'single-stream' - compare with P=1 baseline using parameter-matched R value
    """
    output_path = Path(output_dir)

    # Calculate differences from baseline
    diff_results = {}
    for key, group in df.groupby(level=[0, 1]):
        base_model, p_value = key

        if baseline_mode == 'single-stream':
            baseline_col = (base_model, 'P=1', 'Repro LoRA %s' % BASELINE_RANK[p_value])

        else:
            baseline_col = key + ('Repro LoRA %s' % BASELINE_RANK[p_value],)
            group = group.drop(baseline_col)

        if baseline_col not in df.index:
            logging.warning("Couldn't find %s in DataFrame; skipping relative results for %s", baseline_col, key)
            continue
        diff_results[key] = group - df.loc[baseline_col].values

    diff_df = pd.concat(diff_results.values())

    # Apply same model ordering to relative plots
    if model_ordering is None:
        model_ordering = get_model_ordering(df)
    diff_df = sort_models_within_clusters(diff_df, model_ordering)

    # Set up plotting parameters for differences
    dmax = np.nanmax(np.abs(diff_df.values))
    diff_kwargs = {
        "square": True,
        "annot": True,
        "annot_kws": {"size": 7},
        "fmt": "+.1%",
        "cmap": "bwr_r",
        "vmin": -dmax,
        "vmax": dmax,
        "cbar": False,
    }

    # Get unique base models and P values
    base_models = diff_df.index.get_level_values(0).unique().dropna()
    p_values = diff_df.index.get_level_values(1).unique().dropna()

    if len(base_models) == 0:
        logging.warning("No relative data found for plotting")
        return diff_df

    # Calculate width ratios based on number of columns in each subplot
    widths = []
    for base_model in base_models:
        for p_value in p_values:
            try:
                mask = ((diff_df.index.get_level_values(0) == base_model) &
                        (diff_df.index.get_level_values(1) == p_value))
                subset = diff_df.loc[mask]
                if len(subset) > 0:
                    widths.append(len(subset))
            except (KeyError, IndexError):
                pass

    # Calculate figure size based on total width ratios
    total_width = sum(widths)
    fig_width = max(14, total_width * 0.7)
    fig_height = 12

    fig, axs = plt.subplots(1, len(widths), figsize=(fig_width, fig_height),
                            tight_layout=True, sharey=True,
                            gridspec_kw={'width_ratios': widths})
    if isinstance(axs, mpl.axes._axes.Axes):
        axs = np.array([axs])
    axs = axs.reshape(-1)

    ix = 0
    for ix, (name, subset_df) in enumerate(diff_df.groupby(level=[0, 1])):
        ax = axs[ix]

        sns.heatmap(subset_df.droplevel([0, 1, -1]).T, ax=ax, **diff_kwargs)
        ax.set_title(f"{name[0]} | {name[1]}")
        ax.tick_params(axis='x', labelsize=10, rotation=90)

        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # Save plot
    plot_file = output_path / f'{plot_type}-{analysis_mode}-{baseline_mode}-relative.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info("Generated relative scores plot (%s baseline): %s", baseline_mode, plot_file)
    return diff_df


def generate_model_eval_summaries(diff_df, df_with_metadata, output_dir, plot_type='all', analysis_mode="quick", model_ordering=None, baseline_mode='single-stream'):
    """Generate model-level and eval-level summary statistics with heatmaps."""
    output_path = Path(output_dir)

    # Apply consistent model ordering
    if model_ordering is None:
        model_ordering = get_model_ordering(df_with_metadata)
    sorted_diff_df = sort_models_within_clusters(diff_df, model_ordering)

    # Model-level summary (using sorted data)
    model_summary_df = pd.DataFrame({
        "max_gain_pct": sorted_diff_df.max(axis=1),
        "p80_gain_pct": sorted_diff_df.quantile(0.800, axis=1),
        "p50_gain_pct": sorted_diff_df.quantile(0.500, axis=1),
        "p20_gain_pct": sorted_diff_df.quantile(0.200, axis=1),
        "mean_gain_pct": sorted_diff_df.mean(axis=1),
        "better_pct": (sorted_diff_df > 0).sum(axis=1) / pd.notna(sorted_diff_df).sum(axis=1),
    })

    # Eval-level summary (using sorted data)
    eval_summary_df = pd.DataFrame({
        "max_gain_pct": sorted_diff_df.max(axis=0),
        "p80_gain_pct": sorted_diff_df.quantile(0.800, axis=0),
        "p50_gain_pct": sorted_diff_df.quantile(0.500, axis=0),
        "p20_gain_pct": sorted_diff_df.quantile(0.200, axis=0),
        "mean_gain_pct": sorted_diff_df.mean(axis=0),
        "better_pct": (sorted_diff_df > 0).sum(axis=0) / pd.notna(sorted_diff_df).sum(axis=0),
    }).sort_values("mean_gain_pct", ascending=False)

    vmax = max(eval_summary_df.drop(columns=["better_pct"]).abs().max().max(),
               model_summary_df.drop(columns="better_pct").abs().max().max())
    _plot_summary_heatmap(output_path, prettify_multiindex(model_summary_df.droplevel(level=-1), axis=0),
                          vmax, plot_type, "model", analysis_mode, baseline_mode)
    _plot_summary_heatmap(output_path, eval_summary_df, vmax, plot_type, "eval", analysis_mode, baseline_mode)
    return model_summary_df, eval_summary_df


def _plot_summary_heatmap(output_path, summary_df, vmax, plot_type, summary_type, analysis_mode, baseline_mode='single-stream'):
    n_models = len(summary_df)
    cell_size = 0.8
    fig_height = n_models * cell_size + 2
    fig_width = 3 * cell_size + 12  # 2 cols for gains + 1 col for better_pct + space for labels

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                   sharey=True, tight_layout=True)

    # Plot gain percentages with diverging colormap
    sns.heatmap(summary_df.drop(columns=["better_pct"]), ax=ax1, annot=True, fmt='.1%',
                cmap="RdBu", center=0, vmin=-vmax, vmax=vmax, cbar=False, square=True)

    # Plot better_pct with 0-1 range colormap
    sns.heatmap(summary_df[["better_pct"]], ax=ax2, annot=True, fmt='.1%',
                cmap="Blues", cbar=False, square=True)

    # Ensure all model names are visible
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10, rotation=90)
    ax2.tick_params(axis='x', labelsize=10, rotation=90)

    ax1.set_title(f'{summary_type.title()}-Level Summary ({plot_type.title()}, {baseline_mode})')
    heatmap_file = output_path / f'{plot_type}-{analysis_mode}-{baseline_mode}-{summary_type}.png'
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info("Generated %s summary heatmap: %s", summary_type, heatmap_file)


def filter_dataframe_by_type(df, data_map, plot_type):
    """Filter dataframe based on plot type."""
    if plot_type == 'all':
        return df

    # Create filtered data map
    filtered_map = {}

    for model_name in data_map.keys():
        for dataset_metric in data_map[model_name].keys():
            dataset_name = dataset_metric[0]

            include = False
            if plot_type == 'summ' and ('CNN' in dataset_name or 'XSum' in dataset_name):
                include = True
            elif plot_type == 'qa' and ('TriviaQA' in dataset_name or 'NQ' in dataset_name or 'TruthfulQA' in dataset_name):
                include = True
            elif plot_type == 'instr' and ('MemoTrap' in dataset_name or 'IFEval' in dataset_name):
                include = True
            elif plot_type == 'detect' and ('HaluEval' in dataset_name or 'SelfCheck' in dataset_name):
                include = True
            elif plot_type == 'rc' and ('RACE' in dataset_name or 'SQuAD' in dataset_name):
                include = True
            elif plot_type == "pub":
                include = (all(not re.search(pat, dataset_name) for pat in PUB_EVAL_BLACKLIST) and
                           all(not re.search(pat, model_name) for pat in PUB_MODEL_BLACKLIST))

            if include:
                if dataset_metric not in filtered_map:
                    filtered_map[dataset_metric] = {}
                filtered_map[dataset_metric][model_name] = data_map[model_name][dataset_metric]

    # Convert to DataFrame
    filtered_df = pd.DataFrame.from_dict(filtered_map, orient='index')
    filtered_df.index = [', '.join(map(str, idx)) for idx in filtered_df.index]

    # Filter rows with sufficient data
    counts = filtered_df.count(axis=1)
    if len(counts) > 0:
        filtered_df = filtered_df.reindex(counts[counts >= 0.5 * counts.max()].index)

    # Sort
    filtered_df = filtered_df.sort_index(axis=0, key=lambda x: x.str.lower())
    filtered_df = filtered_df.reindex(sorted(filtered_df.columns), axis=1)

    # Drop word_perplexity metrics
    filtered_df = filtered_df[~filtered_df.index.str.contains('word_perplexity', case=False, na=False)]

    # Set small values to NaN
    filtered_df[filtered_df < 1e-3] = np.nan

    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Analyze ParScale experiments and generate leaderboard plots")
    parser.add_argument("--results-base-path", type=Path, default="leaderboard",
                        help="Path to evaluation results directory")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots and data")
    parser.add_argument("--no-download", dest="download_repos", action="store_false", help="Skip downloading from HF & S3")
    parser.add_argument("--model-whitelist", nargs="+", help="Model name patterns to include",
                        default=[
                            "mistralai/Mistral-7B-v0.1",
                            "meta-llama/Llama-2-7b-hf",
                            "ParControl/",
                            "Qwen/Qwen2.5-0.5B",
                            "Qwen/Qwen2.5-1.5B",
                        ])
    parser.add_argument("--s3-base-path", type=str, default="s3://obviouslywrong-parcontrol/ParControl",
                        help="S3 path for syncing ParControl model results")

    parser.add_argument("--plot-mode", nargs="+", choices=['all', "pub", 'summ', 'qa', 'instr', 'detect', 'rc'],
                        default=['all', "pub"], help="Types of plots to generate")
    parser.add_argument("--analysis-mode", default="full", help="Depth of analysis to run", choices=["quick", "deep", "full"])
    parser.add_argument("--baseline-mode", default="single-stream", choices=["stream-matched", "single-stream"],
                        help="Baseline calculation mode: stream-matched (P=X vs P=X baseline) or single-stream (P=X vs P=1 baseline)")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    args.results_path = args.results_base_path / f"evals-{args.analysis_mode}"
    args.s3_path = args.s3_base_path.rstrip("/") + f"/evals-{args.analysis_mode}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and verify
    df, data_map = parse_leaderboard_results(results_path=args.results_path, model_whitelist=args.model_whitelist,
                                             model_name_mapping=MODEL_NAMES, download_repos=args.download_repos,
                                             s3_path=args.s3_path)
    unmapped = [col for col in df.columns if col.startswith("ParControl/") and col not in MODEL_NAMES]
    assert not unmapped, f"Found {len(unmapped)} models not in MODEL_NAMES: {unmapped}"

    for plot_type in args.plot_mode:
        logging.info("Processing plot type: %s", plot_type)

        # Filter by plot type
        filtered_df = filter_dataframe_by_type(df, data_map, plot_type)
        if filtered_df.empty:
            logging.warning("No data found for plot type: %s", plot_type)
            continue

        metadata = [parse_model_metadata(c) for c in filtered_df.columns]
        filtered_df.columns = pd.MultiIndex.from_tuples([(m[0], m[1], m[2], c) for m, c in zip(metadata, filtered_df.columns)])
        filtered_df.T.sort_index().to_parquet(Path(args.output_dir) / f'{plot_type}-{args.analysis_mode}-{args.baseline_mode}.parquet')
        df_with_metadata = generate_absolute_plots(filtered_df, args.output_dir, plot_type, args.analysis_mode)
        diff_df = generate_relative_plots(df_with_metadata, args.output_dir, plot_type, args.analysis_mode,
                                          baseline_mode=args.baseline_mode)
        generate_model_eval_summaries(diff_df, df_with_metadata, args.output_dir, plot_type, args.analysis_mode,
                                      baseline_mode=args.baseline_mode)


if __name__ == "__main__":
    main()
