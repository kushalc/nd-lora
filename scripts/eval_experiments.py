#!/usr/bin/env python3
"""
Evaluate ND-LoRA checkpoints on Modal, reusing backend_cli functions for code sharing.

Run as: uv run scripts/eval_experiments.py <mode> [--workers N]
"""

import argparse
import logging
import random

import modal

from leaderboard.backend_cli import app, evaluate_all_models, parse_args
from utils.model_checkpoints import ALL_CHECKPOINTS as MODEL_CHECKPOINTS, S3_BUCKET

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

GENERAL_HALLUC_EVAL_SUITE = [
    "wikitext",
    "pile",
    "winogrande",
]

FULL_HALLUC_EVAL_SUITE = [
    "halueval_dialogue",
    "halueval_qa",
    "halueval_summarization",
    "memo-trap_v2",
    "nq_swap",
    "nq8",
    "popqa",
    "tqa8",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
] + GENERAL_HALLUC_EVAL_SUITE

# Evaluation suite / sample-budget presets -> backend_cli args.
EVAL_MODES = {
    "full":    {"tasks": FULL_HALLUC_EVAL_SUITE, "s3": "evals-full"},
    "general": {"tasks": GENERAL_HALLUC_EVAL_SUITE, "s3": "evals-general"},
    "deep":    {"sample_limit": 1024, "s3": "evals-deep"},
    "quick":   {"sample_limit": 128, "s3": "evals-quick"},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate ND-LoRA checkpoints on Modal")
    parser.add_argument("mode", choices=list(EVAL_MODES), help="Evaluation suite / sample budget")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel eval workers to spawn over the checkpoint set")
    args = parser.parse_args(argv)

    spec = EVAL_MODES[args.mode]
    # Randomly permute so multiple workers cover the checkpoint set in parallel.
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    cli = []
    if "tasks" in spec:
        cli += ["--eval-benchmark-tasks"] + spec["tasks"]
    if "sample_limit" in spec:
        cli += [f"--sample-limit={spec['sample_limit']}"]
    cli += [f"--s3-base-dir={S3_BUCKET}/evals/{spec['s3']}"]
    kwargs = vars(parse_args(cli + checkpoints))

    with modal.enable_output(), app.run(detach=True):
        if args.workers == 1:
            evaluate_all_models.remote(**kwargs)
        else:
            for worker in range(args.workers):
                logging.info("Spawning eval worker %d", worker)
                evaluate_all_models.spawn(**kwargs)


if __name__ == "__main__":
    main()
