#!/usr/bin/env python3
"""Generate Table 4 (ablations) D_spec values from evaluation results.

Computes spectral diversity (D_spec) for different architectural variants by analyzing
layer activations from neurodiversity evaluation runs. This script reproduces the exact
D_spec values used in Table 4 of the paper.

D_spec calculation:
- Extracts aggregate layer activations from evaluation pickles
- Computes nanmean across all activation dimensions
- Averages across all evaluation samples

Variants analyzed:
- Standard: Repro LoRA R64 (P=1) → D_spec = None (baseline, no diversity)
- ParScale: SharedLoRA R64 (P=4) → D_spec ≈ 0.999 (minimal diversity)
- ParScale-BT: nOSL SharedLoRA R64 (P=4) → D_spec ≈ 0.998 (slight improvement)
- Indep. LoRA: IndLoRA (P=4) → D_spec ≈ 0.231 (stream-aware without BT)
- ND-LoRA: nOSL IndLoRA (P=4) → D_spec ≈ 0.133 (full method with BT)
"""

import argparse
import logging
import pickle
import re
import subprocess
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_checkpoints import S3_BUCKET


def parse_duration(duration_str: str) -> float:
    """Parse duration string like '1h', '30m', '2d' into seconds."""
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([smhd])$', duration_str.lower())
    assert match, f"Invalid duration format: {duration_str}. Use e.g. '1h', '30m', '2d'"
    value, unit = float(match.group(1)), match.group(2)
    multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    return value * multipliers[unit]


def parse_timestamp_from_path(path: str) -> float:
    """Extract run timestamp from filename like 'task-YYYY-MM-DD-HH-MM-SS-seed.pkl'."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', Path(path).name)
    assert match, f"No timestamp found in filename: {path}"
    dt = datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
    return dt.timestamp()


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


_WARNED_LEGACY_DSPEC = set()

# All dspec variants to extract (in order of preference for display)
DSPEC_VARIANTS = ["dspec_D", "dspec_DF", "dspec_frob", "dspec_cosine"]


def _calculate_dspec_all(x, path: str = None) -> dict:
    """Calculate all available dspec variants from layer activations.

    Returns dict with keys like 'dspec_DF', 'dspec_frob', 'dspec_cosine'.
    Missing variants are set to NaN.
    """
    result = {v: np.nan for v in DSPEC_VARIANTS}

    if isinstance(x, dict):
        # Legacy dict format - only has aggregate layer values
        result["dspec_frob"] = np.nanmean(x["model.model.aggregate_layer"]).mean().item()
    elif isinstance(x, list):
        df = pd.DataFrame(x)

        # Map column names to variant keys
        col_mapping = {
            "original_dspec_D": "dspec_D",
            "original_dspec_DF": "dspec_DF",
            "original_dspec_frob": "dspec_frob",
            "original_dspec_cosine": "dspec_cosine",
            "original_dspec": "dspec_cosine",  # Legacy format was cosine similarity
        }

        for col, variant in col_mapping.items():
            if col in df.columns:
                if col == "original_dspec":
                    warn_key = path or id(x)
                    if warn_key not in _WARNED_LEGACY_DSPEC:
                        _WARNED_LEGACY_DSPEC.add(warn_key)
                        logging.warning("Legacy dspec format detected (original_dspec -> dspec_cosine): %s", path or "<unknown>")
                result[variant] = df[col].mean()
    else:
        raise ValueError(f"Unexpected layer_activations type: {type(x)}")

    return result


def _count_samples(x):
    if isinstance(x, dict):
        return x["model.model.aggregate_layer"].shape[0]
    elif isinstance(x, list):
        return len(x)
    else:
        raise ValueError()


def load_evaluation_results(eval_dir, since_seconds: float = None):
    """Load all evaluation pickle files and extract D_spec values.

    Args:
        eval_dir: Directory containing evaluation .pkl files
        since_seconds: If set, only load files modified within this many seconds

    Returns:
        DataFrame with columns: model, task, dspec_DF, dspec_frob, dspec_cosine, path, etc.
    """
    all_paths = glob(str(eval_dir / "*.pkl"))
    if since_seconds is not None:
        cutoff_time = time.time() - since_seconds
        all_paths = [p for p in all_paths if Path(p).stat().st_mtime >= cutoff_time]
        logging.info("Filtered to %d files modified in last %.0f seconds", len(all_paths), since_seconds)

    assert all_paths, f"No .pkl files found in {eval_dir} (after filtering)"

    concat = []
    for path in all_paths:
        try:
            dt = pickle.load(open(path, "rb"))
            dt["path"] = Path(path).name
            dt["path_mtime"] = datetime.fromtimestamp(Path(path).stat().st_mtime)
            # Identify the model by its clean name embedded in the filename
            # (<task>-<clean-model>-<seed>.pkl); the internal `model` field is a legacy run-id URL.
            dt["model"] = Path(path).stem[len(dt["task"]) + 1:].rsplit("-", 1)[0]
            concat.append(dt)
        except:
            logging.warning("Couldn't load %s", path, exc_info=True)

    TASK_METRICS = {
        "nq8": "exact_match,remove_whitespace",
    }

    def _extract_results(row):
        key = TASK_METRICS.get(row["task"], "acc,none")
        for name, val in row["eval_results"]["results"].items():
            return val[key]

    raw_df = pd.DataFrame(concat)
    raw_df["eval_score"] = raw_df.apply(_extract_results, axis=1)

    # Extract all dspec variants into separate columns
    dspec_all = raw_df.apply(lambda row: _calculate_dspec_all(row["layer_activations"], row["path"]), axis=1)
    dspec_df = pd.DataFrame(dspec_all.tolist(), index=raw_df.index)
    for col in DSPEC_VARIANTS:
        raw_df[col] = dspec_df[col]

    raw_df["n_samples"] = raw_df["layer_activations"].apply(_count_samples)

    # Warn about unstable structure for 11/2025 runs
    nov_2025_mask = raw_df["path_mtime"].dt.strftime("%Y-%m") == "2025-11"
    if nov_2025_mask.any():
        logging.warning("Found %d runs from 11/2025 - dspec structure was unstable during this period, results may be inconsistent",
                        nov_2025_mask.sum())

    # Sort by dspec_D (primary), falling back to dspec_DF then dspec_frob for legacy data
    for sort_col in ["dspec_D", "dspec_DF", "dspec_frob"]:
        if raw_df[sort_col].notna().any():
            break
    raw_df = raw_df.sort_values([sort_col]).reset_index(drop=True)
    return raw_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute D_spec values for Table 4 ablations')

    # Paper Table 4 D (RMS cosine) was computed on the N256n vintage: it is the only suite whose
    # ablation pkls (ParScale-BT, Stream LoRA, Stream LoRA-BT) carry original_dspec_D. The N1024n
    # ablation pkls predate the RMS-cos code and store only mean cosine, so D comes back NaN there.
    parser.add_argument('--eval-dir', type=Path, default='outputs/neurodiversity/evals-N256n',
                        help='Directory containing evaluation .pkl files')
    parser.add_argument("--s3-base-path", type=str, nargs="+", default=[f"{S3_BUCKET}/evals/neurodiversity/evals-N256n"],
                        help="S3 path for syncing model results")
    parser.add_argument("--since", type=str, default=None,
                        help="Only include files modified within this duration (e.g. '1h', '30m', '2d')")
    parser.add_argument("--task-regex", type=str, default=None,
                        help="Filter tasks by regex pattern (e.g. 'nq8|triviaqa')")
    parser.add_argument("--no-sync", action="store_true", help="Skip S3 sync")
    args = parser.parse_args()

    if not args.no_sync:
        logging.info("Sync'ing from S3")
        for s3_base_path in args.s3_base_path:
            subprocess.run(["aws", "s3", "sync", s3_base_path, args.eval_dir], check=True)

    since_seconds = parse_duration(args.since) if args.since else None
    raw_df = load_evaluation_results(args.eval_dir, since_seconds=since_seconds)

    if args.task_regex:
        task_mask = raw_df["task"].str.contains(args.task_regex, regex=True)
        matched_tasks = raw_df.loc[task_mask, "task"].unique()
        assert task_mask.any(), f"No tasks matched regex '{args.task_regex}'. Available: {raw_df['task'].unique().tolist()}"
        logging.info("Filtered to %d tasks matching '%s': %s", len(matched_tasks), args.task_regex, matched_tasks.tolist())
        raw_df = raw_df[task_mask]

    # On conflict, keep the seed that actually carries the paper's D (RMS cosine): each (model, task)
    # has one pkl re-evaluated with the RMS-cos code (dspec_D present) and, for some, an older seed
    # where dspec_D is NaN. Prefer the dspec_D-bearing row, breaking ties by most-recent mtime.
    # (The original relied on mtime alone, which is fragile once files are re-copied and mtimes reset.)
    raw_df["_has_D"] = raw_df["dspec_D"].notna()
    raw_df = (raw_df.sort_values(["model", "task", "_has_D", "path_mtime"], ascending=[True, True, False, False])
                    .drop_duplicates(["model", "task"], keep="first")
                    .drop(columns="_has_D"))
    display_cols = ["model", "task", "path", "path_mtime", "n_samples", "eval_score"] + DSPEC_VARIANTS
    logging.info("Captured %d raw samples:\n%s", len(raw_df),
                 raw_df[display_cols].set_index(["model", "task"]).sort_index().to_string())

    summary_df = raw_df.groupby("model").agg({
        "path_mtime": "mean",
        "n_samples": "sum",
        "eval_score": "mean",
    } | {d: "mean" for d in DSPEC_VARIANTS})
    summary_df["path_mtime"] = summary_df["path_mtime"].dt.round("1s")
    summary_df["n_samples"] = summary_df["n_samples"].astype(pd.Int64Dtype())
    summary_df = summary_df.reindex([
        "ParScale_P4_R64",
        "ParScale-BT_P4",
        "Stream_LoRA_P4",
        "Stream_LoRA-BT_P4",
        "ND-LoRA_P2",
        "ND-LoRA_P4",
        "ND-LoRA_P8",
    ])
    logging.info("Calculated D_spec variants by model:\n%s", summary_df.to_string(na_rep=""))

    ndlora_df = raw_df[raw_df['model'].str.startswith('ND-LoRA')].copy()
    ndlora_df['P'] = ndlora_df['model'].str.extract(r'_P(\d+)')[0].astype(int)
    parquet_df = ndlora_df.groupby(['P', "task"])[DSPEC_VARIANTS + ["eval_score"]].mean()
    output_path = Path("outputs") / "table4_task_level.parquet"
    parquet_df.to_parquet(output_path)
    logging.info("Wrote task-level parquet to %s with shape %s", output_path, parquet_df.shape)
