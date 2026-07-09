#!/usr/bin/env python3
"""Table 4 (causality): artificial corruption of neural diversity establishes statistical causality.

Faithful port of the dose-response analysis in ParControl's ``adhoc/analysis-ablations-diversity.ipynb``
(the ``evals-N128d5-S781`` cell). Fractional stream substitution (``eval_neurodiversity.py
--corruption-mode stream``) perturbs neural diversity by a controlled dose; per-sample paired t-tests
vs. the zero-dose baseline, combined by Fisher meta-analysis across sub-experiments (via
``utils.paired_stats.paired_dose_response_analysis``), give Δ𝒟 / ΔScore / SE / d / p per task.

The reported table is the single operating point the paper uses: ``substitute_fraction = 0.1`` at
``dose_level = 1``, for the three causal tasks (see ``tab:causality``).

Data layout (synced from S3):

    evals-N128d5-S781/<mode>-<run_id>/<dose_level>/<task>-<...>.pkl
    evals-N128d5-S781/<mode>-<run_id>/<dose_level>/metadata.<ts>.json

Each pkl carries per-sample ``layer_activations`` (``original_dspec`` + ``corrupted_dspec``) and the
lm-eval ``eval_results`` (with per-sample ``samples``); each metadata json carries the sweep config
(``substitute_fraction``/``substitute_k_streams`` and the per-dose ``sweep_values``).
"""
import argparse
import json
import logging
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_checkpoints import S3_BUCKET
from utils.paired_stats import extract_per_sample_scores, paired_dose_response_analysis

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
# The paper's operating point: fractional stream substitution at 10% of dimensions, first dose step.
CORRUPTION_FRACTION = 0.1
DOSE_LEVEL = 1
CAUSAL_TASKS = ["halueval_summarization", "memo-trap_v2", "truthfulqa_mc2"]


def _get_dspec(sample: dict):
    """Per-sample original 𝒟, tolerating the two legacy key spellings."""
    return sample.get("original_dspec") or sample.get("original__dspec")


def load_dose_data(eval_dir: Path) -> pd.DataFrame:
    """Load every dose pkl into one row per (run, dose, task, resampling_idx) with dspec_n/dspec_c/score."""
    paths = sorted(eval_dir.glob("**/*.pkl"))
    assert paths, (
        f"No dose pkls under {eval_dir}. Sync them first with "
        f"`aws s3 sync {S3_BUCKET}/evals/neurodiversity/{eval_dir.name} {eval_dir}` (see docstring).")

    rows = []
    for path in paths:
        d = pickle.load(open(path, "rb"))
        # .../<mode>-<run_id>/<dose_level>/<file>.pkl
        parts = str(path).rsplit("/", 3)
        d["dose_level"] = int(parts[-2])
        d["corruption_mode"], d["run_id"] = parts[1].split("-")
        d["dspec_n"] = np.nanmean([_get_dspec(y) for y in d["layer_activations"]])
        d["dspec_c"] = (np.nanmean([y.get("corrupted_dspec") for y in d["layer_activations"]])
                        if d["corruption_mode"] != "n" else np.nan)
        first_task = next(iter(d["eval_results"]["results"]))
        d["eval_score"] = d["eval_results"]["results"][first_task].get("acc,none", np.nan)
        rows.append(d)
    df = pd.DataFrame(rows)
    logger.info("Loaded %d dose pkls: %d runs, doses %s, tasks %s", len(df), df["run_id"].nunique(),
                sorted(df["dose_level"].unique()), sorted(df["task"].unique()))
    return df


def load_metadata(eval_dir: Path) -> pd.DataFrame:
    """One row per (run_id, dose_level) with the sweep config; substitute_fraction is the per-dose value."""
    records = []
    for path in eval_dir.glob("**/metadata.*.json"):
        parts = str(path).rsplit("/", 3)
        _mode, run_id = parts[1].split("-")
        dose_level = int(parts[-2])
        meta = json.load(open(path))
        records.append({
            "run_id": run_id,
            "dose_level": dose_level,
            "seed": meta["seed"],
            "corruption_alpha_max": meta["corruption_alpha_max"],
            "substitute_fraction": meta["substitute_fraction"],
            "substitute_k_streams": meta["substitute_k_streams"],
            # The swept parameter takes its per-dose value; when it IS substitute_fraction this
            # overrides the default above (dict-literal: later key wins), matching the notebook.
            **{meta["sweep_param"]: meta["sweep_values"][dose_level]},
        })
    df = pd.DataFrame(records)
    assert not df.empty, f"No metadata.*.json under {eval_dir}; cannot resolve per-dose substitute_fraction."
    return df


def build_causality_table(dose_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample paired dose-response vs. the zero-dose baseline; select fraction=0.1, dose=1."""
    # Keep only resampling_idx that cover (nearly) all (task, fraction, dose) cells, so the pairing is balanced.
    dose_df = dose_df.copy()
    dose_df["task_dose"] = dose_df[["task", "substitute_fraction", "dose_level"]].apply(tuple, axis=1)
    full = dose_df.groupby("resampling_idx")["task_dose"].nunique()
    keep = full[full >= full.max() - 1].index
    logger.info("Keeping %d of %d resampling_idx (>= %d task_dose cells)",
                len(keep), dose_df["resampling_idx"].nunique(), full.max() - 1)

    samples = extract_per_sample_scores(dose_df[dose_df["resampling_idx"].isin(keep)],
                                        eval_results_col="eval_results", task_col="task")
    samples = samples[(samples["substitute_k_streams"] < 3) & (samples["substitute_fraction"] < 0.5)]

    # Collapse the resampling draws so each run contributes ONE 128-sample sub-experiment (matching the
    # paper's "4 sub-experiments x 128 samples" = N=512): average each doc_id's score/𝒟 across its
    # resamplings, then pair natural-vs-dose on doc_id within (run_id, task).
    keys = ["run_id", "task", "dose_level", "substitute_fraction", "doc_id"]
    samples = samples.groupby(keys, as_index=False).agg(
        eval_score=("eval_score", "mean"), dspec_c=("dspec_c", "mean"),
        substitute_k_streams=("substitute_k_streams", "first"))

    summary = paired_dose_response_analysis(
        samples, baseline_dose=0, group_cols=["run_id", "task"],
        sample_col="eval_score", dspec_col="dspec_c",
        stratify_cols=["task", "substitute_fraction"], meta_method="fisher")

    # Notebook sign convention: report perturbation magnitudes as positive (corruption raises 𝒟, lowers acc).
    summary = summary.rename(columns={"substitute_fraction": "corruption", "Δacc": "Δeval", "SE(Δacc)": "SE(Δeval)"})
    summary[["Δdspec", "Δeval"]] = -summary[["Δdspec", "Δeval"]]
    summary["Sig"] = pd.cut(summary["p"], [0, 1e-3, 1e-2, 5e-2, 1],
                            labels=["***", "**", "*", ""], include_lowest=True)

    idx = pd.IndexSlice
    out = (summary.set_index(["task", "corruption", "dose"]).sort_index()
           .loc[idx[:, CORRUPTION_FRACTION, DOSE_LEVEL],
                ["Δdspec", "Δeval", "SE(Δeval)", "d", "p", "Sig", "N"]]
           .reset_index(level=["corruption", "dose"], drop=True)
           .reindex([t for t in CAUSAL_TASKS]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=BASE_DIR / "outputs/neurodiversity/evals-N128d5-S781",
                        help="Local dir for the dose-response suite (synced from S3)")
    parser.add_argument("--s3-base-path", type=str,
                        default=f"{S3_BUCKET}/evals/neurodiversity/evals-N128d5-S781",
                        help="S3 path for the dose-response suite")
    parser.add_argument("--no-sync", action="store_true", help="Skip S3 sync")
    args = parser.parse_args()

    if not args.no_sync:
        subprocess.run(["aws", "s3", "sync", args.s3_base_path, str(args.eval_dir)], check=True)

    dose_df = load_dose_data(args.eval_dir)
    dose_df = dose_df.merge(load_metadata(args.eval_dir), on=["run_id", "dose_level"], how="left")
    assert dose_df["substitute_fraction"].notna().all(), "Missing metadata for some (run_id, dose_level) rows"

    table = build_causality_table(dose_df)
    logger.info("Table 4 (causality) @ fraction=%.2f dose=%d:\n%s",
                CORRUPTION_FRACTION, DOSE_LEVEL, table.to_string(float_format=lambda v: f"{v:.4g}"))
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
