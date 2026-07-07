#!/usr/bin/env python3
"""
Evaluate ParControl experiments using Modal for distributed execution.
Reuses backend_cli.py functions for maximum code sharing.
"""

import logging
import random
import time

from leaderboard.backend_cli import app, evaluate_all_models, parse_args
from utils.model_checkpoints import ALL_CHECKPOINTS as MODEL_CHECKPOINTS, S3_BUCKET

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


@app.local_entrypoint()
def modal__test():
    kwargs = vars(parse_args([
        "--sample-limit=5",
        "Qwen/Qwen2.5-0.5B",
        f"{S3_BUCKET}/checkpoints/2025-09-14-15-20-01",
    ]))
    evaluate_all_models.remote(**kwargs)


GENERAL_HALLUC_EVAL_SUITE = [
    "wikitext",
    "pile",
    "winogrande",
    # "race_4",  # Note: Task not currently supported
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


@app.local_entrypoint()
def modal__fParControl():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args(["--eval-benchmark-tasks"] + FULL_HALLUC_EVAL_SUITE + GENERAL_HALLUC_EVAL_SUITE +
                             [f"--s3-base-dir={S3_BUCKET}/evals/evals-full"] +
                             checkpoints))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__gParControl():
    """Evaluate ParControl models on general purpose evaluation suite only (wikitext, pile, winogrande)."""
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args(["--eval-benchmark-tasks"] + GENERAL_HALLUC_EVAL_SUITE +
                             [f"--s3-base-dir={S3_BUCKET}/evals/evals-general"] +
                             checkpoints))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__dParControl():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args([
        "--sample-limit=1024",
        f"--s3-base-dir={S3_BUCKET}/evals/evals-deep",
    ] + checkpoints))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__dParControl_spawns():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))

    PARALLEL_WORKER_COUNT = 3

    kwargs = vars(parse_args([
        "--sample-limit=1024",
        f"--s3-base-dir={S3_BUCKET}/evals/evals-deep",
    ] + checkpoints))

    for worker in range(PARALLEL_WORKER_COUNT):
        try:
            logging.info(f"Firing worker {worker} to handle checkpoints: {checkpoints}")
            evaluate_all_models.spawn(**kwargs)
            time.sleep(10)
        except Exception as e:
            logging.error("Failed to spawn worker %s on checkpoints %s", worker, checkpoints, exc_info=e)
            continue


@app.local_entrypoint()
def modal__qParControl():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args([
        "--sample-limit=128",
        f"--s3-base-dir={S3_BUCKET}/evals/evals-quick",
    ] + checkpoints))
    evaluate_all_models.remote(**kwargs)
