import logging
import os
import random
import sys

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from src.backend.huggingface_generate_until import HFLMwithChatTemplate
from src.backend.manage_requests import EvalRequest
from src.backend.tasks.cnndm.task import CNNDM
from src.backend.tasks.cnndm.task_v2 import CNNDMv2
from src.backend.tasks.xsum.task import XSum
from src.backend.tasks.xsum.task_v2 import XSumv2

# from src.backend.tasks.selfcheckgpt.task import SelfCheckGPT


# Note: ParScale models are now self-contained with auto_map, no registration needed

def run_evaluation(eval_request: EvalRequest, task_names, num_fewshot, batch_size, device, use_cache=None,
                   limit=None, max_nb_samples=100, seed: int = 42) -> dict:
    # Set random seeds for intra-task inter-model consistency.
    logging.info("Setting random seed: %d for task: %s", seed, task_names)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logging.info("Allocating task manager for: %s", task_names)
    tasks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
    task_manager = TaskManager(include_path=tasks_path)

    model, tokenizer = eval_request.get_model()

    eval_kwargs = {
        "tasks": task_names,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "use_cache": use_cache,
        "cache_requests": False,
        "write_out": False,
        "task_manager": task_manager
    }
    if limit:
        eval_kwargs["samples"] = samples_dict = {}
        task_dict = task_manager.load_task_or_group(task_names)
        for task_name in task_dict:
            total_examples = len(task_dict[task_name].eval_docs)
            samples_dict[task_name] = sorted(random.sample(range(total_examples), min(limit, total_examples)))
            logging.info("Sampled %d random (of %d) examples for task=%s: %s",
                         len(samples_dict[task_name]), total_examples, task_name, samples_dict[task_name])

    if model is not None:
        # Path A: S3/ParScale models (loaded model/tokenizer)
        assert tokenizer is not None, "S3 model loaded but tokenizer is None"
        logging.info("Using loaded S3 model for evaluation")
        eval_kwargs["model"] = HFLM(pretrained=model, tokenizer=tokenizer, dtype=eval_request.precision, device=device,
                                    trust_remote_code=True, backend="causal")
    else:
        # Path B: HuggingFace models - direct evaluation
        logging.info("Using direct HF evaluation for: %s", eval_request.model)
        eval_kwargs["model"] = "hf"
        eval_kwargs["model_args"] = f"pretrained={eval_request.model},dtype={eval_request.precision},device={device},trust_remote_code=True,parallelize=True,device_map=auto"

    results = evaluator.simple_evaluate(**eval_kwargs)
    results["config"]["model_dtype"] = eval_request.precision
    results["config"]["model_name"] = eval_request.model
    results["config"]["model_sha"] = eval_request.revision

    if max_nb_samples is not None:
        if 'samples' in results:
            samples = results['samples']
            for task_name in samples.keys():
                if len(samples[task_name]) > max_nb_samples:
                    results['samples'][task_name] = results['samples'][task_name][:max_nb_samples]

    return results
