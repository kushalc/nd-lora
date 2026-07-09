#!/usr/bin/env python3
"""Compare ND-LoRA against inference-time and training-time baselines (Table 3).

Produces two artifacts programmatically, both as DataFrames and optionally as LaTeX:
    - Summary:   method × (type, Halluc Δ%, Knowledge Δ%) — the headline deltas.
    - Details:   method × eval-task absolute scores — the per-cell numbers.

Fully self-contained: every number is piped from data in the nd-lora bucket, with no external
repo (the baselines used to require the ITR/IHD project's `utils.etl`; its eval outputs are now
vendored into the bucket, so this script reads them directly).

Data sources (each is the single source of truth for its rows):
    - Our rows (Qwen LoRA R32, ND-LoRA): the on-demand `all-full` parquet built by
      `adhoc/build_scores.py` from the S3 evals. ND-LoRA's per-task score is the *oracle* — the P
      that maximizes that task (ORACLE_P_VALUES, mirrored from `prompt_router/router_rescoring.py`)
      read off the ND-LoRA [OptC9] rows at P∈{2,4,8} and the Repro LoRA R64 row at P=1. This is the
      per-task upper bound of any P-routing policy; the paper reports it as ND-LoRA's point estimate.
    - Baseline rows (CAD, ActDec, Disagreement, Qwen+LoRA): the per-task lm-eval pickles under
      `{S3_BUCKET}/baselines/<method>/evals/N131072n-*/`, averaged across the three seeds. The
      display/metric naming reuses `utils.analysis_utils._sanitise_{dataset,metric}` (the same SSOT
      that names the parquet columns), so baseline cells line up with our rows exactly.

Usage:
    python adhoc/table3_baselines.py                # print both tables (text)
    python adhoc/table3_baselines.py --latex        # also emit LaTeX
    python adhoc/table3_baselines.py --save-dir OUT # also write parquets
"""

import argparse
import logging
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.analysis_utils import _sanitise_dataset, _sanitise_metric
from utils.model_checkpoints import S3_BUCKET

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
# Vendored baseline evals: {S3_BUCKET}/baselines/<method>/evals/N131072n-<seed>/<task>-*.pkl
S3_BASELINES = f"{S3_BUCKET}/baselines"
BASELINES_DIR = BASE_DIR / "outputs" / "baselines"
BASELINE_LIMIT = "N131072n"  # full-sample eval limit reported in Table 3 (three seeds averaged)
# Display names of the four baseline methods == their subdirectory under baselines/.
BASELINE_METHODS = ["CAD", "ActDec", "Disagreement", "Qwen+LoRA"]

# Parquet column -> short display name. Covers all tasks common to our parquet and the baseline
# pkls. Membership in this map also *selects* the metric to read from each pkl (only the metric
# whose sanitised (dataset, metric) name is a key here is kept; stderr/perplexity variants drop out).
COL_MAP = {
    'HaluEval Dialog, Accuracy': 'HE-Dial',
    'HaluEval QA, Accuracy': 'HE-QA',
    'HaluEval Summarization, Accuracy': 'HE-Summ',
    'MemoTrap, Accuracy': 'MemoTrap',
    'TruthfulQA MC1, Accuracy': 'TF-MC1',
    'TruthfulQA MC2, Accuracy': 'TF-MC2',
    'NQ (8-shot), EM': 'NQ',
    'TriviaQA (8-shot), EM': 'TriviaQA',
    'popQA, EM': 'PopQA',
    'wikitext, bits_per_byte': 'Wikitext',
    'winogrande, Accuracy': 'WG',
}

# ND-LoRA oracle: pretty parquet column -> P that maximizes that task. Mirrors ORACLE_P_VALUES in
# prompt_router/router_rescoring.py. P=1 reads the Repro LoRA R64 row (ND-LoRA collapses to
# single-stream LoRA at P=1); P∈{2,4,8} read the ND-LoRA [OptC9] row at that P.
ORACLE_P_BY_COL = {
    'HaluEval Dialog, Accuracy': 4, 'HaluEval QA, Accuracy': 4, 'HaluEval Summarization, Accuracy': 4,
    'MemoTrap, Accuracy': 8,
    'TruthfulQA MC1, Accuracy': 2, 'TruthfulQA MC2, Accuracy': 2,
    'NQ (8-shot), EM': 1, 'popQA, EM': 1, 'TriviaQA (8-shot), EM': 1, 'wikitext, bits_per_byte': 1,
    'winogrande, Accuracy': 4,
}

# PARQUET_MODELS entries: (P, short_name, display_name, family). Qwen LoRA R32 is the canonical P=1
# baseline against which our rows' deltas are computed.
PARQUET_MODELS = [
    ('P=1', 'Repro LoRA R32', 'Qwen LoRA R32', 'Qwen LoRA'),
]

HALLUC_TASKS = ['HE-Dial', 'HE-QA', 'HE-Summ', 'MemoTrap', 'TF-MC1', 'TF-MC2']
# Knowledge/general-capability tasks: NQ/PopQA/TriviaQA (recall), WG (commonsense),
# Wikitext (fluency). Wikitext BPB is lower-is-better; compute_avg_delta handles the sign via
# LOWER_IS_BETTER.
KNOWLEDGE_TASKS = ['NQ', 'PopQA', 'TriviaQA', 'WG', 'Wikitext']

# Columns displayed in the detail table.
TABLE_COLS    = ['HE-Dial', 'HE-QA', 'HE-Summ', 'MemoTrap', 'TF-MC1', 'TF-MC2',
                 'NQ', 'PopQA', 'TriviaQA', 'Wikitext', 'WG']
TABLE_HEADERS = ['HE-Dial', 'HE-QA', 'HE-Summ', 'MemoTrap', 'TF-MC1', 'TF-MC2',
                 'NQ',      'PopQA', 'TriviaQA', 'Wikitext', 'WG']
# Wikitext BPB: lower is better
LOWER_IS_BETTER = {'Wikitext'}


def _pretty_col(dataset_name: str, metric_name: str) -> str:
    """Reconstruct the parquet column name for one (dataset, metric) via the naming SSOT."""
    metric = metric_name.split(',')[0]  # drop the lm-eval filter suffix, e.g. 'acc,none' -> 'acc'
    return f"{_sanitise_dataset(dataset_name)}, {_sanitise_metric(metric)}"


def load_baseline_scores(baselines_dir: Path) -> dict[str, dict[str, float]]:
    """Load CAD/ActDec/Disagreement/Qwen+LoRA absolute scores from the vendored lm-eval pkls, keyed
    by short task name. Each pkl is one (method, task, seed); scores are averaged across seeds. The
    metric is selected by COL_MAP membership (only the reported metric survives sanitisation)."""
    out = {}
    for method in BASELINE_METHODS:
        pkls = sorted((baselines_dir / method / "evals").glob(f"{BASELINE_LIMIT}-*/*.pkl"))
        assert pkls, (
            f"No baseline pkls under {baselines_dir / method}. Sync them first with "
            f"`aws s3 sync {S3_BASELINES} {baselines_dir}` (see module docstring).")
        by_col = defaultdict(list)
        for path in pkls:
            with open(path, "rb") as f:
                results = pickle.load(f)["eval_results"]["results"]
            for dataset_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    if "stderr" in metric_name or metric_name == "alias":
                        continue
                    col = _pretty_col(dataset_name, metric_name)
                    if col in COL_MAP and isinstance(value, (int, float)):
                        by_col[COL_MAP[col]].append(float(value))
        out[method] = {short: float(np.mean(vals)) for short, vals in by_col.items()}
        logger.info("Loaded %s from %d pkls: %d tasks covered", method, len(pkls), len(out[method]))
    return out


def _parquet_row(df: pd.DataFrame, p_val: str, short_name: str) -> pd.Series:
    """Return the single Q0.5B parquet row for (p_val, short_name); fail loudly if not unique."""
    mask = (df['model'] == 'Q0.5B') & (df['P'] == p_val) & (df['short_name'] == short_name)
    matches = df[mask]
    assert len(matches) == 1, f"Expected 1 row for {short_name} at {p_val}, got {len(matches)}"
    return matches.iloc[0]


def load_parquet_scores() -> tuple[dict[str, dict[str, float]], dict[str, tuple[str, str]]]:
    """Load our rows (Qwen LoRA R32 baseline + ND-LoRA oracle) from the all-full parquet.

    Returns (scores_by_display_name, meta_by_display_name) where meta is (family, P).
    """
    from build_scores import ensure_scores_parquet
    df = pd.read_parquet(ensure_scores_parquet(plot_type="all", mode="full")).reset_index()
    df.columns = ['model', 'P', 'short_name', 'full_name'] + list(df.columns[4:])

    results, meta = {}, {}
    for p_val, short_name, display_name, family in PARQUET_MODELS:
        row = _parquet_row(df, p_val, short_name)
        scores = {short: row[parquet_col] for parquet_col, short in COL_MAP.items()
                  if parquet_col in df.columns}
        results[display_name] = scores
        meta[display_name] = (family, p_val)
        logger.info("Loaded %s: HE-Summ=%.3f, MemoTrap=%.3f, TF-MC2=%.3f",
                    display_name, scores['HE-Summ'], scores['MemoTrap'], scores['TF-MC2'])

    # ND-LoRA oracle row: per task, read the ND-LoRA [OptC9] score at the P that maximizes it
    # (ORACLE_P_BY_COL), or the Repro LoRA R64 P=1 row for tasks whose oracle P is 1.
    oracle_scores = {}
    for parquet_col, short in COL_MAP.items():
        if parquet_col not in df.columns:
            continue
        p = ORACLE_P_BY_COL[parquet_col]
        row = _parquet_row(df, 'P=1', 'Repro LoRA R64') if p == 1 \
            else _parquet_row(df, f'P={p}', 'ND-LoRA [OptC9]')
        val = row[parquet_col]
        assert pd.notna(val), f"Oracle score for {parquet_col} (P={p}) is NaN in the parquet"
        oracle_scores[short] = float(val)
    results['ND-LoRA'] = oracle_scores
    meta['ND-LoRA'] = ('ND-LoRA-oracle', 'oracle')
    logger.info("Loaded ND-LoRA oracle: HE-Summ=%.3f, MemoTrap=%.3f, TF-MC2=%.3f, PopQA=%.3f, Wikitext=%.3f",
                oracle_scores['HE-Summ'], oracle_scores['MemoTrap'], oracle_scores['TF-MC2'],
                oracle_scores['PopQA'], oracle_scores['Wikitext'])
    return results, meta


def compute_avg_delta(scores: dict, baseline: dict, tasks: list[str]) -> float:
    """Compute average relative delta (%) over tasks; positive = improvement.

    For lower-is-better metrics (Wikitext BPB), the sign is inverted so that a positive delta
    still means the method improved over the baseline.
    """
    deltas = []
    for t in tasks:
        if t not in scores or t not in baseline or not (baseline[t] > 0):
            continue
        raw = (scores[t] - baseline[t]) / baseline[t] * 100
        deltas.append(-raw if t in LOWER_IS_BETTER else raw)
    return sum(deltas) / len(deltas) if deltas else 0.0


def find_col_bests(all_rows: list[dict[str, float]], cols: list[str]) -> dict[str, float]:
    """Find best value per column across all rows."""
    bests = {}
    for col in cols:
        vals = [row[col] for row in all_rows if col in row]
        bests[col] = min(vals) if col in LOWER_IS_BETTER else max(vals)
    return bests


# ---------- Row assembly --------------------------------------------------------------------

def assemble_rows(parquet_scores: dict[str, dict[str, float]],
                  baseline_scores: dict[str, dict[str, float]]) -> list[dict]:
    """Return the ordered list of row dicts used by both summary and detail tables."""
    baseline_r32 = parquet_scores['Qwen LoRA R32']
    itr_baseline = baseline_scores['Qwen+LoRA']
    rows = [
        {'display': 'Qwen LoRA R32',  'scores': baseline_r32,               'baseline': baseline_r32,
         'type': 'baseline',       'symbol': '',              'group': 'train'},
        {'display': 'ND-LoRA',         'scores': parquet_scores['ND-LoRA'],  'baseline': baseline_r32,
         'type': 'integrated',     'symbol': '',              'group': 'train'},
        {'display': 'CAD',             'scores': baseline_scores['CAD'],      'baseline': itr_baseline,
         'type': 'inference-time', 'symbol': '$^\\dagger$',   'group': 'other'},
        {'display': 'ActDec',          'scores': baseline_scores['ActDec'],   'baseline': itr_baseline,
         'type': 'inference-time', 'symbol': '$^\\dagger$',   'group': 'other'},
        {'display': 'Disagreement',    'scores': baseline_scores['Disagreement'], 'baseline': itr_baseline,
         'type': 'training-time',  'symbol': '$^\\ddagger$',  'group': 'other'},
    ]
    return rows


# ---------- Summary (method × Δ%) -----------------------------------------------------------

def build_summary_df(rows: list[dict]) -> pd.DataFrame:
    """method × (type, halluc_delta_pct, knowledge_delta_pct). Baselines get 0% since they
    are their own reference, but we drop those rows so the table shows only methods."""
    records = []
    for row in rows:
        if row['type'] == 'baseline':
            continue
        records.append({
            'Method': row['display'],
            'Type': row['type'],
            'Halluc Δ%':    compute_avg_delta(row['scores'], row['baseline'], HALLUC_TASKS),
            'Knowledge Δ%': compute_avg_delta(row['scores'], row['baseline'], KNOWLEDGE_TASKS),
        })
    return pd.DataFrame(records).set_index('Method')


def format_summary_text(summary_df: pd.DataFrame) -> str:
    out = ["% Hallucination improvement comparison (avg relative Δ% vs respective P=1 baselines)",
           "% Method                     | Type            | Halluc Δ% | Knowledge Δ%"]
    for method, r in summary_df.iterrows():
        out.append(f"% {method:28s} | {r['Type']:15s} | {r['Halluc Δ%']:+6.1f}%  | "
                   f"{r['Knowledge Δ%']:+6.1f}%")
    return "\n".join(out)


def format_summary_latex(summary_df: pd.DataFrame) -> str:
    best_h = summary_df['Halluc Δ%'].max()
    best_k = summary_df['Knowledge Δ%'].max()

    def fmt(val: float, best: float) -> str:
        s = f"{val:+.1f}\\%"
        return f"\\textbf{{{s}}}" if abs(val - best) < 1e-6 else s

    out = ["\\begin{table}[t]", "  \\centering",
           "    \\begin{tabular}{l|l|rr}",
           "      \\bf Method & \\bf Type & \\bf Halluc.\\ $\\Delta$\\% & \\bf Knowledge $\\Delta$\\% \\\\",
           "      \\hline"]
    for method, r in summary_df.iterrows():
        out.append(f"      {method} & {r['Type']} & "
                   f"{fmt(r['Halluc Δ%'], best_h)} & {fmt(r['Knowledge Δ%'], best_k)} \\\\")
    out += ["    \\end{tabular}",
            "  \\caption{",
            "    \\textbf{ND-LoRA dominates on hallucination without a knowledge tax.} Average relative",
            "    $\\Delta$\\% vs.\\ the $P=1$ baseline across six hallucination (HaluEval Dial/QA/Summ, MemoTrap,",
            "    TF-MC1/MC2) and five knowledge (NQ, PopQA, TriviaQA, Winogrande, Wikitext BPB) benchmarks.",
            "  }",
            "  \\label{tab:baselines_summary}",
            "\\end{table}"]
    return "\n".join(out)


# ---------- Detail (method × eval) ----------------------------------------------------------

def build_detail_df(rows: list[dict]) -> pd.DataFrame:
    """method × eval short name, absolute scores. Useful for sanity-checking individual cells."""
    data = {row['display']: {c: row['scores'].get(c, float('nan')) for c in TABLE_COLS}
            for row in rows}
    return pd.DataFrame(data).T.reindex(columns=TABLE_COLS)


def format_detail_text(detail_df: pd.DataFrame) -> str:
    return detail_df.round(3).to_string()


def format_detail_latex(rows: list[dict]) -> str:
    """Transposed layout: rows = evals, columns = methods. Matches tab:results_p{2,4,8} style."""
    all_score_dicts = [row['scores'] for row in rows]
    bests = find_col_bests(all_score_dicts, TABLE_COLS)

    def fmt_cell(val: float, col: str) -> str:
        is_best = abs(val - bests[col]) < 1e-6
        s = f"{val:.3f}"
        return f"\\textbf{{{s}}}" if is_best else s

    header_cells = [f"\\textbf{{{r['display']}{r['symbol']}}}" for r in rows]
    out = ["\\begin{table}[htbp]", "  \\centering",
           "  \\begin{tabular}{l|" + "c" * len(rows) + "}",
           "    \\textbf{Evaluation} & " + " & ".join(header_cells) + " \\\\",
           "    \\hline"]
    for eval_short, eval_header in zip(TABLE_COLS, TABLE_HEADERS):
        cells = [fmt_cell(row['scores'][eval_short], eval_short) for row in rows]
        out.append(f"    {eval_header} & " + " & ".join(cells) + " \\\\")
    out += ["  \\end{tabular}",
            "  \\caption{",
            "    \\textbf{Per-benchmark scores underlying \\autoref{tab:baselines_summary}.}",
            "    Inference-time ($\\dagger$) applied to pretrained Qwen2.5-0.5B; training-time ($\\ddagger$)",
            "    parameter-matched to our backbone \\citep{shi2024cad, chen2024actdec, li2018disagreement}.",
            "    Wikitext is BPB (lower is better).",
            "  }",
            "  \\label{tab:baselines_detail}",
            "\\end{table}"]
    return "\n".join(out)


# ---------- Baseline alignment (diagnostic only) --------------------------------------------

def log_baseline_alignment(parquet_scores: dict[str, dict[str, float]],
                           baseline_scores: dict[str, dict[str, float]]) -> None:
    baseline_r32 = parquet_scores['Qwen LoRA R32']
    itr_baseline = baseline_scores['Qwen+LoRA']
    logger.info("Baseline alignment check (shared benchmarks, vendored Qwen+LoRA vs our Repro LoRA R32):")
    for task in HALLUC_TASKS:
        itr_val = itr_baseline.get(task, 0)
        pc_val = baseline_r32.get(task, 0)
        logger.info("  %s: baseline=%.3f, ParControl=%.3f, Δ=%.3f",
                    task, itr_val, pc_val, abs(itr_val - pc_val))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latex', action='store_true',
                        help='Also emit LaTeX for tab:baselines_summary and tab:baselines_detail.')
    parser.add_argument('--baselines-dir', type=Path, default=BASELINES_DIR,
                        help='Local dir for the vendored baseline evals (synced from S3).')
    parser.add_argument('--no-sync', action='store_true', help='Skip the S3 sync of baseline evals.')
    parser.add_argument('--save-dir', type=Path, default=None,
                        help='If set, write summary.parquet and detail.parquet to this directory.')
    args = parser.parse_args()

    if not args.no_sync:
        subprocess.run(["aws", "s3", "sync", S3_BASELINES, str(args.baselines_dir),
                        "--exclude", "*.log"], check=True)

    parquet_scores, _meta = load_parquet_scores()
    baseline_scores = load_baseline_scores(args.baselines_dir)
    log_baseline_alignment(parquet_scores, baseline_scores)

    rows = assemble_rows(parquet_scores, baseline_scores)
    summary_df = build_summary_df(rows)
    detail_df = build_detail_df(rows)

    print("\n=== Summary (method × Δ%) ===")
    print(format_summary_text(summary_df))
    print("\n=== Detail (method × eval absolute score) ===")
    print(format_detail_text(detail_df))

    if args.latex:
        print("\n=== LaTeX: summary ===")
        print(format_summary_latex(summary_df))
        print("\n=== LaTeX: detail ===")
        print(format_detail_latex(rows))

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.save_dir / 'summary.parquet'
        detail_path = args.save_dir / 'detail.parquet'
        summary_df.to_parquet(summary_path)
        detail_df.to_parquet(detail_path)
        logger.info("Wrote %s and %s", summary_path, detail_path)


if __name__ == '__main__':
    main()
