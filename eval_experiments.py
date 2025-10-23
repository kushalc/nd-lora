#!/usr/bin/env python3
"""
Evaluate ParControl experiments using Modal for distributed execution.
Reuses backend_cli.py functions for maximum code sharing.
"""

import logging
import random
import time

from leaderboard.backend_cli import app, evaluate_all_models, parse_args
from utils.model_checkpoints_paper import BASE_CHECKPOINTS, ALL_CHECKPOINTS as MODEL_CHECKPOINTS

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


@app.local_entrypoint()
def modal__test():
    kwargs = vars(parse_args([
        "--sample-limit=5",
        "Qwen/Qwen2.5-0.5B",
        "s3://obviouslywrong-parcontrol/ParControl/2025-09-14-15-20-01",
    ]))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__Qwen():
    kwargs = vars(parse_args([
        # "--batch-size=4",  # Note: Default 32 breaks SQuaDv2 and a few others for Qwen, override for 2nd run
        # "--s3-base-dir=s3://obviouslywrong/ParControl/evals/2025-09-15-09-09-04",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
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
    # checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()) + BASE_CHECKPOINTS,
    #                             len(MODEL_CHECKPOINTS) + len(BASE_CHECKPOINTS))
    # Note: Using only ParControl models for evaluation
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args(["--eval-benchmark-tasks"] + FULL_HALLUC_EVAL_SUITE + GENERAL_HALLUC_EVAL_SUITE +
                             ["--s3-base-dir=s3://obviouslywrong-parcontrol/ParControl/evals-full"] +
                             checkpoints))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__gParControl():
    """Evaluate ParControl models on general purpose evaluation suite only (wikitext, pile, winogrande)."""
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args(["--eval-benchmark-tasks"] + GENERAL_HALLUC_EVAL_SUITE +
                             ["--s3-base-dir=s3://obviouslywrong-parcontrol/ParControl/evals-general"] +
                             checkpoints))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__dParControl():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    # checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()) + BASE_CHECKPOINTS,
    #                             len(MODEL_CHECKPOINTS) + len(BASE_CHECKPOINTS))
    # Note: Using only ParControl models for evaluation
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args([
        "--sample-limit=1024",
        "--s3-base-dir=s3://obviouslywrong-parcontrol/ParControl/evals-deep",

    ] + checkpoints))
    evaluate_all_models.remote(**kwargs)


@app.local_entrypoint()
def modal__dParControl_spawns():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    # checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()) + BASE_CHECKPOINTS,
    #                             len(MODEL_CHECKPOINTS) + len(BASE_CHECKPOINTS))
    # Note: Using only ParControl models for evaluation
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))

    PARALLEL_WORKER_COUNT = 3

    kwargs = vars(parse_args([
        "--sample-limit=1024",
        "--s3-base-dir=s3://obviouslywrong-parcontrol/ParControl/evals-deep",
    ] + checkpoints))

    for worker in range(PARALLEL_WORKER_COUNT):
        try:
            logging.info(f"Firing worker {worker} to handle checkpoints: {checkpoints}")
            evaluate_all_models.spawn(**kwargs)
            time.sleep(10)
        except Exception as e:
            logging.error(f"Failed to spawn worker: {worker} on checkpoints {checkpoints}", e)
            continue


@app.local_entrypoint()
def modal__qParControl():
    # NOTE: Randomly permute so we can have multiple workers running effectively in parallel
    # checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()) + BASE_CHECKPOINTS,
    #                             len(MODEL_CHECKPOINTS) + len(BASE_CHECKPOINTS))
    # Note: Using only ParControl models for evaluation
    checkpoints = random.sample(list(MODEL_CHECKPOINTS.values()), len(MODEL_CHECKPOINTS))
    kwargs = vars(parse_args([
        "--sample-limit=128",
        "--s3-base-dir=s3://obviouslywrong-parcontrol/ParControl/evals-quick",
    ] + checkpoints))
    evaluate_all_models.remote(**kwargs)
