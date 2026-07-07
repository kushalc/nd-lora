#!/usr/bin/env python3
"""
LoRA Rank Confound Analysis

Generates:
1. Figure: Performance vs rank at equal scaling (α=2r) - shows rank doesn't determine performance
2. Table: Decomposition comparing α=2r vs α=32 across ranks

Usage:
    python adhoc/table10_lora_confounders.py [--no-ci]
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs/assets"
# Aggregated 'all'-cohort parquets are regenerated on demand from raw S3 evals (see build_scores).
# NOTE: parquet index model names come from analyze_experiments.parse_model_metadata() /
# utils.model_checkpoints.MODEL_NAMES on the eval config.model_name — verify after first build.
from build_scores import ensure_scores_parquet
PARQUET_DEEP = ensure_scores_parquet(plot_type="all", mode="deep")
PARQUET_FULL = ensure_scores_parquet(plot_type="all", mode="full")
# JSON stderr-fallback dirs: analyze_experiments syncs raw evals here during the build above.
EVALS_DEEP_DIR = BASE_DIR / "outputs/evals-deep/ParControl"

# JSON stderr fallback: value = eval-dir name under P=1/. LoRA-ablation sweep kept its run-ids;
# the Repro baselines were renamed to clean model names during the bucket migration.
JSON_FALLBACK_RUNS = {
    "LoRA ablation R2": "2025-11-26-00-30-03",
    "LoRA ablation R4": "2025-11-26-00-30-17",
    "LoRA ablation R8": "2025-11-26-00-30-33",
}

JSON_FULL_RUNS = {
    "LoRA ablation R16": "2025-11-26-00-10-20",
    "LoRA ablation R32a": "2025-11-26-00-34-01",
    "LoRA ablation R64a": "2025-11-26-00-34-07",
    "LoRA ablation R128a": "2025-11-26-00-34-27",
    "Repro LoRA R32": "Qwen2.5-0.5B_P1_R32",
    "Repro LoRA R64": "Qwen2.5-0.5B_P1_R64",
    "Repro LoRA R128": "Qwen2.5-0.5B_P1_R128",
}
EVALS_FULL_DIR = BASE_DIR / "outputs/evals-full/ParControl"

# Reverse mapping: parquet column name -> (JSON task name, metric key, stderr key)
PARQUET_TO_JSON_TASK = {
    'HaluEval Dialog, Accuracy': ('halueval_dialogue', 'acc,none', 'acc_stderr,none'),
    'HaluEval QA, Accuracy': ('halueval_qa', 'acc,none', 'acc_stderr,none'),
    'HaluEval Summarization, Accuracy': ('halueval_summarization', 'acc,none', 'acc_stderr,none'),
    'MemoTrap, Accuracy': ('memo-trap_v2', 'acc,none', 'acc_stderr,none'),
    'NQ (8-shot), EM': ('nq8', 'exact_match,remove_whitespace', 'exact_match_stderr,remove_whitespace'),
    'nq swap, EM': ('nq_swap', 'exact_match,remove_whitespace', 'exact_match_stderr,remove_whitespace'),
    'popQA, EM': ('popqa', 'exact_match,remove_whitespace', 'exact_match_stderr,remove_whitespace'),
    'TriviaQA (8-shot), EM': ('tqa8', 'exact_match,remove_whitespace', 'exact_match_stderr,remove_whitespace'),
    'TruthfulQA MC1, Accuracy': ('truthfulqa_mc1', 'acc,none', 'acc_stderr,none'),
    'TruthfulQA MC2, Accuracy': ('truthfulqa_mc2', 'acc,none', 'acc_stderr,none'),
    'wikitext, bits_per_byte': ('wikitext', 'bits_per_byte,none', 'bits_per_byte_stderr,none'),
    'winogrande, Accuracy': ('winogrande', 'acc,none', 'acc_stderr,none'),
}

# Model name mappings for parquet lookup
# α=2r experiments (scaling = 2.0 for all) - named "LoRA ablation RX" in parquet
ALPHA_2R_MODELS = {
    2: "LoRA ablation R2",
    4: "LoRA ablation R4",
    8: "LoRA ablation R8",
    16: "LoRA ablation R16",
    32: "LoRA ablation R32a",
    64: "LoRA ablation R64a",
    128: "LoRA ablation R128a",
}

# α=32 experiments - named "Repro LoRA RX" in parquet
# R16 α=32 comes from α=2r experiments (since 2*16=32)
ALPHA_32_MODELS = {
    16: "LoRA ablation R16",  # α=2*16=32, so α/r=2
    32: "Repro LoRA R32",
    64: "Repro LoRA R64",
    128: "Repro LoRA R128",
}

# ND-LoRA models by P value
ND_LORA_MODELS = {
    2: ("P=2", "ND-LoRA [OptC9]"),
    4: ("P=4", "ND-LoRA [OptC9]"),
    8: ("P=8", "ND-LoRA [OptC9]"),
}

# Parquet column name -> short display name mapping (matching Tables 7-9)
HALLUCINATION_COLS = {
    'HaluEval Dialog, Accuracy': 'HE Dialog',
    'HaluEval QA, Accuracy': 'HE QA',
    'HaluEval Summarization, Accuracy': 'HE Summ',
    'MemoTrap, Accuracy': 'MemoTrap',
    'TruthfulQA MC1, Accuracy': 'TF-MC1',
    'TruthfulQA MC2, Accuracy': 'TF-MC2',
}

OTHER_COLS = {
    'NQ (8-shot), EM': 'NQ-8',
    'nq swap, EM': 'NQ-swap',
    'popQA, EM': 'PopQA',
    'TriviaQA (8-shot), EM': 'TQA-8',
    'wikitext, bits_per_byte': 'Wikitext BPB',
    'winogrande, Accuracy': 'Winogrande',
}

ALL_COLS = {**HALLUCINATION_COLS, **OTHER_COLS}
HALLUCINATION_COL_NAMES = list(HALLUCINATION_COLS.keys())


def load_parquet_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare parquet data for deep and full evals."""
    df_deep = pd.read_parquet(PARQUET_DEEP).reset_index()
    df_deep.columns = ['model', 'P', 'short_name', 'full_name'] + list(df_deep.columns[4:])

    df_full = pd.read_parquet(PARQUET_FULL).reset_index()
    df_full.columns = ['model', 'P', 'short_name', 'full_name'] + list(df_full.columns[4:])

    return df_deep, df_full


def load_stderr_from_json(run_dir: Path) -> dict:
    """Load stderr values from JSON files in a run directory."""
    json_files = sorted(run_dir.glob("results_*.json"), reverse=True)
    if not json_files:
        return {}

    results = {}
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        for task_name, task_data in data.get('results', {}).items():
            for parquet_col, (json_task, metric_key, stderr_key) in PARQUET_TO_JSON_TASK.items():
                if task_name == json_task:
                    short_name = ALL_COLS[parquet_col]
                    if stderr_key in task_data:
                        results[f'{short_name}_stderr'] = task_data[stderr_key]
    return results


def load_from_json_fallback(model_name: str, p_value: str = "P=1") -> dict | None:
    """Load results from JSON files. Checks evals-full first, then evals-deep."""
    if model_name in JSON_FULL_RUNS:
        run_id = JSON_FULL_RUNS[model_name]
        run_dir = EVALS_FULL_DIR / p_value / run_id
    elif model_name in JSON_FALLBACK_RUNS:
        run_id = JSON_FALLBACK_RUNS[model_name]
        run_dir = EVALS_DEEP_DIR / p_value / run_id
    else:
        return None

    assert run_dir.exists(), f"Run dir not found: {run_dir}"
    json_files = sorted(run_dir.glob("results_*.json"), reverse=True)
    assert json_files, f"No JSON files in {run_dir}"

    results = {}
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        for task_name, task_data in data.get('results', {}).items():
            # Find matching parquet column
            for parquet_col, (json_task, metric_key, stderr_key) in PARQUET_TO_JSON_TASK.items():
                if task_name == json_task:
                    short_name = ALL_COLS[parquet_col]
                    if metric_key in task_data:
                        results[short_name] = task_data[metric_key]
                    if stderr_key in task_data:
                        results[f'{short_name}_stderr'] = task_data[stderr_key]
    return results if results else None


def load_models_by_name(df: pd.DataFrame, model_mapping: dict, p_value: str = "P=1",
                        use_full_evals: bool = False) -> pd.DataFrame:
    """Load results for models by their short_name from parquet DataFrame, with JSON fallback.

    Args:
        use_full_evals: If True, load stderr from evals-full directory (for α=32 models)
    """
    rows = []
    evals_dir = EVALS_FULL_DIR if use_full_evals else EVALS_DEEP_DIR
    json_runs = JSON_FULL_RUNS if use_full_evals else JSON_FALLBACK_RUNS

    for rank, model_name in model_mapping.items():
        mask = (df['P'] == p_value) & (df['short_name'] == model_name)
        matches = df[mask]

        if len(matches) == 0:
            # Try JSON fallback
            json_results = load_from_json_fallback(model_name, p_value)
            if json_results:
                logger.warning("Model %s not in parquet, using JSON fallback (run analyze_experiments.py to fix)",
                               model_name)
                rows.append({'rank': rank, **json_results})
                logger.info("Loaded R%d (%s) from JSON: %d metrics", rank, model_name, len(json_results))
            else:
                logger.warning("Model not found: %s (%s)", model_name, p_value)
            continue

        row_data = {'rank': rank}
        for col in ALL_COLS.keys():
            if col in matches.columns:
                row_data[ALL_COLS[col]] = matches[col].iloc[0]

        # Load stderr from JSON (not in parquet)
        run_id = json_runs.get(model_name)
        if run_id:
            run_dir = evals_dir / p_value / run_id
            stderr_data = load_stderr_from_json(run_dir)
            row_data.update(stderr_data)

        rows.append(row_data)
        logger.info("Loaded R%d (%s): %d metrics", rank, model_name, len(row_data) - 1)

    assert len(rows) > 0, f"No results loaded for {p_value}"
    return pd.DataFrame(rows)


def load_nd_lora_from_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Load ND-LoRA results from parquet."""
    rows = []
    for p_val, (p_str, model_name) in ND_LORA_MODELS.items():
        mask = (df['P'] == p_str) & (df['short_name'] == model_name)
        matches = df[mask]
        if len(matches) == 0:
            logger.warning("ND-LoRA not found: %s (P=%s)", model_name, p_str)
            continue

        row_data = {'P': p_val}
        for col in ALL_COLS.keys():
            if col in matches.columns:
                row_data[ALL_COLS[col]] = matches[col].iloc[0]
        rows.append(row_data)
        logger.info("Loaded ND-LoRA P=%d: %d metrics", p_val, len(row_data) - 1)

    return pd.DataFrame(rows) if rows else None


def compute_hallucination_average(df: pd.DataFrame) -> pd.Series:
    """Compute average across hallucination tasks."""
    short_names = list(HALLUCINATION_COLS.values())
    available = [n for n in short_names if n in df.columns]
    assert len(available) >= 3, f"Too few hallucination tasks found: {available}"
    return df[available].mean(axis=1)


def compute_hallucination_stderr(df: pd.DataFrame) -> pd.Series:
    """Compute pooled stderr for hallucination average (sqrt of mean of variances)."""
    stderr_cols = [f'{name}_stderr' for name in HALLUCINATION_COLS.values()]
    available = [c for c in stderr_cols if c in df.columns]
    if not available:
        return pd.Series([np.nan] * len(df), index=df.index)
    # Pooled stderr = sqrt(mean of variances) = sqrt(mean of stderr^2)
    variances = df[available].apply(lambda x: x**2 if isinstance(x.iloc[0], (int, float)) else np.nan)
    return np.sqrt(variances.mean(axis=1))


def generate_figure(df_2r: pd.DataFrame, df_32: pd.DataFrame, df_nd: pd.DataFrame = None):
    """Generate Figure: Rank vs Performance at Equal Scaling, with ND-LoRA for scale."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    df_2r = df_2r.copy()
    df_32 = df_32.copy()
    df_2r['hallu_avg'] = compute_hallucination_average(df_2r)
    df_32['hallu_avg'] = compute_hallucination_average(df_32)
    df_2r['hallu_stderr'] = compute_hallucination_stderr(df_2r)
    df_32['hallu_stderr'] = compute_hallucination_stderr(df_32)

    df_2r = df_2r.sort_values('rank')
    df_32 = df_32.sort_values('rank')

    # Plot α=2r (equal scaling) with error bars
    ax.errorbar(df_2r['rank'], df_2r['hallu_avg'], yerr=1.96*df_2r['hallu_stderr'],
                fmt='o-', linewidth=2, markersize=8, capsize=3,
                label=r'Single LoRA, $\alpha/r=2$', color='C0')

    # Plot α=32 (varying scaling) with error bars
    ax.errorbar(df_32['rank'], df_32['hallu_avg'], yerr=1.96*df_32['hallu_stderr'],
                fmt='s--', linewidth=2, markersize=8, capsize=3,
                label=r'Single LoRA, $\alpha=32$', color='C1')

    # Plot ND-LoRA for comparison (shows scale of improvement)
    if df_nd is not None and len(df_nd) > 0:
        df_nd = df_nd.copy()
        df_nd['hallu_avg'] = compute_hallucination_average(df_nd)
        # ND-LoRA uses R16 per stream, so total rank = P * 16
        df_nd['total_rank'] = df_nd['P'] * 16
        df_nd = df_nd.sort_values('P')

        # Plot line connecting ND-LoRA points (no stderr available from parquet)
        ax.plot(df_nd['total_rank'], df_nd['hallu_avg'], '*-', linewidth=2, markersize=15,
                color='C2', zorder=5, label='ND-LoRA (P×R16)', markeredgecolor='black', markeredgewidth=0.5)

        # Annotate ND-LoRA points
        for _, row in df_nd.iterrows():
            ax.annotate(f"P={int(row['P'])}", (row['total_rank'], row['hallu_avg']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xscale('log', base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])

    ax.set_xlabel('LoRA Rank (or Total Rank for ND-LoRA)', fontsize=11)
    ax.set_ylabel('Avg. Hallucination Accuracy (6 tasks)', fontsize=11)
    ax.set_title('LoRA Rank Does Not Determine Hallucination Performance', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)

    # Y-axis: show useful precision (e.g., 40.0%, 42.5%, 45.0%)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.set_ylim(0.38, 0.52)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / 'lora_rank_confound.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'lora_rank_confound.png', dpi=300, bbox_inches='tight')
    logger.info("Saved figure to %s", OUTPUT_DIR / 'lora_rank_confound.pdf')
    plt.close(fig)


def generate_tables(df_2r: pd.DataFrame, df_32: pd.DataFrame, show_ci: bool = True, show_low_rank: bool = False):
    """Generate two transposed LaTeX tables: one for α=32, one for α/r=2."""
    df_2r = df_2r.copy()
    df_32 = df_32.copy()

    df_2r['scaling'] = 2.0
    df_32['scaling'] = 32.0 / df_32['rank']

    # Metrics using short names (column names in DataFrame)
    hallu_metrics = list(HALLUCINATION_COLS.values())  # ['HE Dial', 'HE QA', 'HE Sum', 'Memo', 'MC1', 'MC2']
    other_metrics = list(OTHER_COLS.values())  # ['Wiki', 'NQ8', 'TQA8', 'Wino', 'NQ-Sw', 'PopQA']

    hline = "[-0.75em]\\\\ \\hline \\\\[-0.75em]"

    def _value(df_sorted, r, metric):
        if r not in df_sorted['rank'].values or metric not in df_sorted.columns:
            return "-"
        row = df_sorted[df_sorted['rank'] == r].iloc[0]
        v = row[metric]
        if pd.isna(v):
            return "-"
        stderr_col = f'{metric}_stderr'
        if show_ci and stderr_col in df_sorted.columns:
            se = row[stderr_col]
            # Handle 'N/A' strings and other non-numeric values
            if not pd.isna(se) and isinstance(se, (int, float)):
                ci = 1.96 * se  # 95% CI
                return f"{v:.2f}\\tiny{{$\\pm${ci:.2f}}}"
        return f"{v:.3f}"

    def print_transposed_table(df, ranks, caption, label, scaling_row=True):
        """Print transposed table with metrics as rows, ranks as columns."""
        df_sorted = df.sort_values('rank')
        n_cols = len(ranks) + 1  # metric name + ranks
        col_spec = 'l' + 'c' * len(ranks)

        print("\\begin{table}[htbp]")
        print("  \\centering")
        print("  \\small")
        print(f"  \\begin{{tabular}}{{{col_spec}}}")

        # Header row with ranks
        rank_headers = [f"R{r}" for r in ranks]
        print(f"    Metric & {' & '.join(rank_headers)} \\\\")
        print(f"    {hline}")

        # Hallucination metrics
        for metric in hallu_metrics:
            vals = [_value(df_sorted, r, metric) for r in ranks]
            print(f"    {metric} & {' & '.join(vals)} \\\\")

        print(f"    {hline}")

        # Other metrics
        for metric in other_metrics:
            vals = [_value(df_sorted, r, metric) for r in ranks]
            print(f"    {metric} & {' & '.join(vals)} \\\\")

        if scaling_row:
            print(f"    {hline}")
            scaling_vals = [f"{df_sorted[df_sorted['rank'] == r]['scaling'].iloc[0]:.2f}"
                            if r in df_sorted['rank'].values else "-" for r in ranks]
            print(f"    $\\alpha/r$ & {' & '.join(scaling_vals)} \\\\")

        print(f"    {hline}")
        print("  \\end{tabular}")
        print(f"  \\caption{{{caption}}}")
        print(f"  \\label{{{label}}}")
        print("\\end{table}")

    # Table 1: α=32 (constant alpha, varying scaling)
    print("\n% LaTeX Table: Constant Alpha (α=32)")
    print_transposed_table(df_32, [16, 32, 64, 128],
                           "Constant $\\alpha=32$: scaling varies with rank. Performance is flat despite 4$\\times$ rank increase.",
                           "tab:lora_constant_alpha")

    # Table 2: α/r=2 (constant scaling)
    # Default to R16-R128 (8× variation); R2/R4/R8 excluded due to post-training retention confound
    scaling_ranks = [2, 4, 8, 16, 32, 64, 128] if show_low_rank else [16, 32, 64, 128]
    rank_variation = "64" if show_low_rank else "8"
    rank_range = "R2$\\rightarrow$R128" if show_low_rank else "R16$\\rightarrow$R128"
    print("\n% LaTeX Table: Constant Scaling (α/r=2)")
    print_transposed_table(df_2r, scaling_ranks,
                           f"Constant scaling $\\alpha/r=2$: performance is flat across {rank_variation}$\\times$ rank variation ({rank_range}), showing rank/expressivity is not the driver.",
                           "tab:lora_constant_scaling")


def main():
    parser = argparse.ArgumentParser(description='LoRA Rank Confound Analysis')
    parser.add_argument('--no-ci', action='store_true', help='Disable confidence intervals in tables')
    parser.add_argument('--show-low-rank', action='store_true',
                        help='Include R2/R4/R8 in constant-scaling table (excluded by default due to post-training retention)')
    args = parser.parse_args()

    logger.info("Starting LoRA rank confound analysis")

    # Load data from parquet files
    df_deep, df_full = load_parquet_data()
    logger.info("Loaded parquet: %d deep rows, %d full rows", len(df_deep), len(df_full))

    # Load model results (with stderr from JSON)
    df_2r = load_models_by_name(df_deep, ALPHA_2R_MODELS, p_value="P=1", use_full_evals=False)
    df_32 = load_models_by_name(df_full, ALPHA_32_MODELS, p_value="P=1", use_full_evals=True)
    df_nd = load_nd_lora_from_parquet(df_full)

    logger.info("Loaded %d α=2r models, %d α=32 models, %d ND-LoRA models",
                len(df_2r), len(df_32), len(df_nd) if df_nd is not None else 0)

    # Generate outputs
    generate_figure(df_2r, df_32, df_nd)
    generate_tables(df_2r, df_32, show_ci=not args.no_ci, show_low_rank=args.show_low_rank)

    # Summary statistics
    df_2r['hallu_avg'] = compute_hallucination_average(df_2r)
    range_2r = df_2r['hallu_avg'].max() - df_2r['hallu_avg'].min()
    logger.info("α=2r performance range across 64x rank variation: %.1f%% (R2=%.1f%%, R128=%.1f%%)",
                range_2r * 100, df_2r[df_2r['rank'] == 2]['hallu_avg'].iloc[0] * 100,
                df_2r[df_2r['rank'] == 128]['hallu_avg'].iloc[0] * 100)


if __name__ == '__main__':
    main()
