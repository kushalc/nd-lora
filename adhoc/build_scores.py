#!/usr/bin/env python3
"""On-demand builder for the aggregated single-stream score parquets.

The figure/table scripts consume ``plots/<plot_type>-<mode>-single-stream.parquet``
(e.g. ``plots/pub-full-single-stream.parquet``). Rather than committing those
artifacts, we regenerate them from the raw per-sample eval JSON that lives in
``s3://obviouslywrong-ndlora/evals/evals-<mode>/`` by driving ``analyze_experiments.py``
(which syncs the raw evals and writes the parquet at ``analyze_experiments.py:460``).

Usage (import):
    from build_scores import ensure_scores_parquet
    path = ensure_scores_parquet(plot_type="pub", mode="full")   # -> Path to parquet
"""
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "plots"
RESULTS_BASE = REPO_ROOT / "leaderboard"           # analyze reads results_base_path/evals-<mode>
S3_EVALS_BASE = "s3://obviouslywrong-ndlora/evals"
BASELINE_MODE = "single-stream"


def scores_parquet_path(plot_type: str = "pub", mode: str = "full") -> Path:
    """Canonical location analyze_experiments writes to (see analyze_experiments.py:460)."""
    return PLOTS_DIR / f"{plot_type}-{mode}-{BASELINE_MODE}.parquet"


def sync_raw_evals(mode: str = "full") -> Path:
    """Sync the raw per-model eval JSON for one suite from S3 (paper-repro subset).

    Excludes the large all_results_*.json aggregates — analyze/statsig/figure1 read the
    per-model results_*.json, which already carry the aggregate `results` dict.
    Returns the local results_path analyze_experiments expects.
    """
    dst = RESULTS_BASE / f"evals-{mode}"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", f"{S3_EVALS_BASE}/evals-{mode}", str(dst),
           "--exclude", "*all_results_*"]
    logging.info("Syncing raw evals: %s -> %s (excluding all_results)", cmd[3], dst)
    subprocess.run(cmd, check=True)
    return dst


def ensure_scores_parquet(plot_type: str = "pub", mode: str = "full", force: bool = False) -> Path:
    """Return the aggregated score parquet, regenerating it from raw S3 evals if absent.

    Fails loudly if analyze_experiments does not produce the expected file.
    """
    out = scores_parquet_path(plot_type, mode)
    if out.exists() and not force:
        logging.info("Using cached scores parquet: %s", out)
        return out

    sync_raw_evals(mode)
    logging.info("Regenerating scores parquet from raw evals: %s (mode=%s)", out, mode)
    cmd = [
        sys.executable, str(REPO_ROOT / "analyze_experiments.py"),
        "--analysis-mode", mode,
        "--plot-mode", plot_type,
        "--baseline-mode", BASELINE_MODE,
        "--output-dir", str(PLOTS_DIR),
        "--results-base-path", str(RESULTS_BASE),
        "--no-download",   # S3 already synced above; skip HF leaderboard snapshot
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    assert out.exists(), f"analyze_experiments did not produce expected parquet: {out}"
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-type", default="pub")
    ap.add_argument("--mode", default="full", choices=["quick", "deep", "full"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(ensure_scores_parquet(a.plot_type, a.mode, a.force))
