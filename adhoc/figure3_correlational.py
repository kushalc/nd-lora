#!/usr/bin/env python3
"""Figure 3: neural diversity vs. HaluEval-Summarization performance for ND-LoRA P=4.

Regresses per-eval-run spectral diversity against task accuracy across the ~39 resampling draws of
ND-LoRA (P=4) on HaluEval-Summarization, reproducing ``fig:correlational``: diversity is negatively
correlated with accuracy (paper: slope=-37.842, R²=0.237, p=0.002).

Metric: the paper's x-axis is the legacy ``dspec_cosine`` (per-sample ``original_dspec`` averaged per
pkl; range ~0.366-0.369). These vintage pkls do NOT carry ``dspec_D`` (RMS-cosine), which is why the
Neural Diversity Index column is NaN here and cosine is the only usable diversity value.

Data: the ``evals-N128`` neurodiversity suite is a single-model point-cloud — one pkl per resampling
draw of ND-LoRA_P4 on halueval_summarization. Each pkl -> one ``dspec_cosine`` (via the SSOT
``table5_dspec_ablations._calculate_dspec_all``) paired with one ``acc,none``. The folder is already
filtered to that (model, task), so no model-name mapping is needed. Fails loudly (rather than fitting
a degenerate line) if fewer than ``--min-points`` draws are present.
"""
import argparse
import logging
import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from table5_dspec_ablations import _calculate_dspec_all
from utils.model_checkpoints import S3_BUCKET

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs/assets"
DIVERSITY_COL = "dspec_cosine"   # the legacy 𝒟 the paper's Figure 3 regresses (see module docstring)
TASK_METRICS = {"nq8": "exact_match,remove_whitespace"}  # default is 'acc,none'


def _eval_score(eval_results: dict, task: str) -> float:
    key = TASK_METRICS.get(task, "acc,none")
    for _name, val in eval_results["results"].items():
        return val[key]


def load_pointcloud(eval_dir: Path) -> pd.DataFrame:
    """One row per pkl: (task, DIVERSITY_COL, eval_score). The dir is pre-filtered to one (model, task)."""
    paths = sorted(eval_dir.glob("*.pkl"))
    assert paths, (
        f"No pkls under {eval_dir}. Sync them first with "
        f"`aws s3 sync {S3_BUCKET}/evals/neurodiversity/{eval_dir.name} {eval_dir}` (see docstring).")
    rows = []
    for path in paths:
        d = pickle.load(open(path, "rb"))
        dspec = _calculate_dspec_all(d["layer_activations"], path.name)
        rows.append({"task": d["task"], DIVERSITY_COL: dspec[DIVERSITY_COL],
                     "eval_score": _eval_score(d["eval_results"], d["task"]), "path": path.name})
    df = pd.DataFrame(rows)
    logger.info("Loaded %d pkls across tasks %s", len(df), sorted(df["task"].unique()))
    return df


def generate_correlational_plot(df, task: str, output_dir: Path, min_points: int) -> dict:
    """Fit and plot DIVERSITY_COL vs. eval_score for one task; fail loudly on too few points."""
    sel = df[df["task"] == task].dropna(subset=[DIVERSITY_COL, "eval_score"])
    assert len(sel) >= min_points, (
        f"Only {len(sel)} {task} points with {DIVERSITY_COL} in the suite; need >= {min_points}. "
        f"Copy the full evals-N128 draw set (see module docstring), then re-run."
    )

    slope, intercept, r_value, p_value, std_err = stats.linregress(sel[DIVERSITY_COL], sel["eval_score"])
    logger.info("Regression (%d points): slope=%.3f R²=%.3f p=%.3e", len(sel), slope, r_value ** 2, p_value)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(sel, x=DIVERSITY_COL, y="eval_score", ax=ax)
    ax.text(0.635, 0.95, f"y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value ** 2:.3f}\np = {p_value:.1e}",
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=18, verticalalignment='top')
    ax.grid(which="both", axis="both", ls=":", lw=1, alpha=0.75)
    ax.set_xlabel(r"$\mathcal{D}_{\text{spec}}$", fontsize=20)
    ax.set_ylabel("HaluEval-Summarization", fontsize=20)
    ax.set_title("ND-LoRA Neural Diversity vs. HaluEval-Summarization", fontsize=22, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(output_dir / f"figure3_correlational.{ext}", dpi=600, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure to %s/figure3_correlational.*", output_dir)
    return {"slope": slope, "intercept": intercept, "r_squared": r_value ** 2,
            "p_value": p_value, "std_err": std_err, "n": len(sel)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=BASE_DIR / "outputs/neurodiversity/evals-N128",
                        help="Local dir for the single-model draw set (synced from S3)")
    parser.add_argument("--s3-base-path", type=str, default=f"{S3_BUCKET}/evals/neurodiversity/evals-N128",
                        help="S3 path for the single-model draw set")
    parser.add_argument("--task", default="halueval_summarization", help="Task to correlate")
    parser.add_argument("--min-points", type=int, default=30, help="Fail if fewer than this many draws")
    parser.add_argument("--no-sync", action="store_true", help="Skip S3 sync")
    args = parser.parse_args()

    if not args.no_sync:
        subprocess.run(["aws", "s3", "sync", args.s3_base_path, str(args.eval_dir)], check=True)

    df = load_pointcloud(args.eval_dir)
    stats_out = generate_correlational_plot(df, args.task, OUTPUT_DIR, args.min_points)
    logger.info("Figure 3 correlation: %s", stats_out)


if __name__ == "__main__":
    main()
