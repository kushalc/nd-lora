#!/usr/bin/env python

import argparse
import json
import logging
import os
import pprint
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import modal
import pandas as pd
import yaml
from botocore.exceptions import ClientError

from src.backend.envs import DEVICE, EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND, LIMIT, Task, Tasks
from src.backend.manage_requests import EvalRequest, check_completed_evals, get_eval_requests, set_eval_request
from src.backend.run_eval_suite import run_evaluation
from src.backend.sort_queue import sort_models_by_priority
from src.envs import API, QUEUE_REPO, RESULTS_REPO
from src.leaderboard.read_evals import EvalResult, get_raw_eval_results
from src.utils import get_tasks_by_benchmarks, my_snapshot_download
from utils.checkpoint_utils import upload_to_s3


def my_set_eval_request(api, eval_request, set_to_status, hf_repo, local_dir):
    for i in range(10):
        try:
            set_eval_request(api=api, eval_request=eval_request, set_to_status=set_to_status, hf_repo=hf_repo, local_dir=local_dir)
            return
        except Exception:
            time.sleep(60)
    return


# Modal setup
# MODAL_GPU = "A10"  # Use A100s for fp32 Q1.5B; A10s break.
MODAL_GPU = "A100-80GB"
# MODAL_GPU = "H200" 
MODAL_IMAGE = modal.Image.debian_slim(python_version="3.11") \
    .pip_install_from_requirements(Path(__file__).parent / "requirements.txt") \
    .pip_install_from_requirements(Path(__file__).parent.parent / "requirements.txt") \
    .run_commands("python -m spacy download en_core_web_sm") \
    .env({"TOKENIZERS_PARALLELISM": "false"}) \
    .add_local_dir(Path(__file__).parent / "src", "/root/src") \
    .add_local_dir(Path(__file__).parent / "cli", "/root/cli") \
    .add_local_dir(Path(__file__).parent.parent / "utils", "/root/utils") \
    .add_local_dir(Path(__file__).parent.parent / "ParScale", "/root/ParScale")
app = modal.App("ParControl")

logging.getLogger("openai").setLevel(logging.WARNING)
pp = pprint.PrettyPrinter(width=80)

PENDING_STATUS = "PENDING"
RUNNING_STATUS = "RUNNING"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"

TASKS_HARNESS = [task.value for task in Tasks]


def check_s3_results_exist(s3_base_dir: str, model_alias: str, task_name: str) -> bool:
    """Check if evaluation results already exist in S3. If task_name provided, check task-level completion."""
    if not s3_base_dir or not s3_base_dir.startswith('s3://'):
        return None

    # Parse S3 path
    path_parts = s3_base_dir.split('/')
    if len(path_parts) < 3:
        return None

    bucket_name = path_parts[2]
    s3_key_prefix = '/'.join(path_parts[3:]) + f"/{model_alias}/"

    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_key_prefix
        )

        if 'Contents' not in response:
            return None

        for obj in response['Contents']:
            key = obj['Key']
            # Check individual task result files (results_YYYY-MM-DD-HH-MM-SS.json)
            if "results_" not in key or not key.endswith('.json'):
                continue

            try:
                result_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = result_obj['Body'].read().decode('utf-8')
                result_data = json.loads(content)
                s3_path = f"s3://{bucket_name}/{key}"

                config = result_data.get('config', {})
                if task_name in config.get('task_name', '') or task_name in str(config):
                    return s3_path
                elif 'results' in result_data and task_name in str(result_data):
                    return s3_path
                elif task_name in result_data and isinstance(result_data[task_name], dict) and 'error' not in result_data[task_name]:
                    return s3_path
            except:
                logging.warning("Couldn't read results: %s", key, exc_info=True)
                continue

        return None

    except ClientError as e:
        logging.warning("Failed to check S3 for existing results: %s", e)
        return None
    except Exception as e:
        logging.warning("Unexpected error checking S3: %s", e)
        return None


# Initialize repositories only in standard mode
def initialize_repositories():
    my_snapshot_download(repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
    my_snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)


def sanity_checks():
    logging.info("Device: %s", DEVICE)

    # pull the eval dataset from the hub and parse any eval requests
    # check completed evals and set them to finished
    my_snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
    check_completed_evals(api=API, checked_status=RUNNING_STATUS, completed_status=FINISHED_STATUS,
                          failed_status=FAILED_STATUS, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND,
                          hf_repo_results=RESULTS_REPO, local_dir_results=EVAL_RESULTS_PATH_BACKEND)
    return


def request_to_result_name(request: EvalRequest) -> str:
    # Request: EvalRequest(model='meta-llama/Llama-2-13b-hf', private=False, status='FINISHED',
    # json_filepath='./eval-queue-bk/meta-llama/Llama-2-13b-hf_eval_request_False_False_False.json',
    # weight_type='Original', model_type='pretrained', precision='float32', base_model='', revision='main',
    # submitted_time='2023-09-09T10:52:17Z', likes=389, params=13.016, license='?')
    #
    # EvalResult(eval_name='meta-llama_Llama-2-13b-hf_float32', full_model='meta-llama/Llama-2-13b-hf',
    # org='meta-llama', model='Llama-2-13b-hf', revision='main',
    # results={'nq_open': 33.739612188365655, 'triviaqa': 74.12505572893447},
    # precision=<Precision.float32: ModelDetails(name='float32', symbol='')>,
    # model_type=<ModelType.PT: ModelDetails(name='pretrained', symbol='🟢')>,
    # weight_type=<WeightType.Original: ModelDetails(name='Original', symbol='')>,
    # architecture='LlamaForCausalLM', license='?', likes=389, num_params=13.016, date='2023-09-09T10:52:17Z', still_on_hub=True)
    #
    org_and_model = request.model.split("/", 1)
    if len(org_and_model) == 1:
        model = org_and_model[0]
        res = f"{model}_{request.precision}"
    else:
        org = org_and_model[0]
        model = org_and_model[1]
        res = f"{org}_{model}_{request.precision}"
    return res


def process_evaluation(
    task: Task,
    eval_request: EvalRequest,
    *,
    upload_to_hub: bool = True,
    batch_size: str = "auto",
    model_alias=None,
    sample_limit: int = LIMIT,
    sync_to_s3: bool = False,
    s3_base_dir: str = None,
    seed: int = 42,
) -> dict:
    # Load model directly
    eval_request.load_model()

    results = run_evaluation(
        eval_request=eval_request,
        task_names=[task.benchmark],
        num_fewshot=task.num_fewshot,
        batch_size=batch_size,
        device=DEVICE,
        use_cache=None,
        limit=sample_limit,
        seed=seed,
        max_nb_samples=None,
    )
    if not model_alias:
        model_alias = eval_request.model
    results["config"]["model_name"] = model_alias
    dumped = json.dumps(results, indent=2, default=lambda o: '<not serializable>')

    now = pd.Timestamp.now(tz="US/Pacific").strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(EVAL_RESULTS_PATH_BACKEND, *model_alias.split("/"), f"results_{now}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dumped)

    if upload_to_hub:
        my_snapshot_download(repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
        API.upload_file(path_or_fileobj=output_path, path_in_repo=f"{model_alias}/results_{now}.json",
                        repo_id=RESULTS_REPO, repo_type="dataset")

    # Upload to S3 if requested
    if sync_to_s3:
        upload_to_s3(output_path, s3_base_dir + f"/{model_alias}", None, logging.getLogger(__name__))

    return results


def process_finished_requests(thr: int, hard_task_lst: Optional[list[str]] = None) -> bool:
    sanity_checks()

    current_finished_status = [FINISHED_STATUS, FAILED_STATUS]

    # Get all eval request that are FINISHED, if you want to run other evals, change this parameter
    eval_requests: list[EvalRequest] = get_eval_requests(job_status=current_finished_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)
    # Sort the evals by priority (first submitted, first run)
    eval_requests: list[EvalRequest] = sort_models_by_priority(api=API, models=eval_requests)

    random.shuffle(eval_requests)

    eval_results: list[EvalResult] = get_raw_eval_results(EVAL_RESULTS_PATH_BACKEND, EVAL_REQUESTS_PATH_BACKEND)

    result_name_to_request = {request_to_result_name(r): r for r in eval_requests}
    result_name_to_result = {r.eval_name: r for r in eval_results}

    for eval_request in eval_requests:
        if eval_request.likes >= thr:
            result_name: str = request_to_result_name(eval_request)

            # Check the corresponding result
            eval_result: Optional[EvalResult] = result_name_to_result[result_name] if result_name in result_name_to_result else None

            # breakpoint()

            # TODO: Configure individual task parameters, don't need to update because only
            # `run_standard_mode` goes through this path.
            task_lst = TASKS_HARNESS.copy()
            # random.shuffle(task_lst)

            # Iterate over tasks and, if we do not have results for a task, run the relevant evaluations
            for task in task_lst:
                task_name = task.benchmark

                do_run_task = False
                if hard_task_lst is None or any(ss in task_name for ss in hard_task_lst):
                    do_run_task = True

                if (eval_result is None or task_name not in eval_result.results) and do_run_task:
                    eval_request: EvalRequest = result_name_to_request[result_name]

                    my_snapshot_download(repo_id=QUEUE_REPO, revision="main",
                                         local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
                    my_set_eval_request(api=API, eval_request=eval_request, set_to_status=RUNNING_STATUS,
                                        hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)

                    results = process_evaluation(task, eval_request, sync_to_s3=False)

                    my_snapshot_download(repo_id=QUEUE_REPO, revision="main",
                                         local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
                    my_set_eval_request(api=API, eval_request=eval_request, set_to_status=FINISHED_STATUS,
                                        hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)

                    return True

    return False


def maybe_refresh_results(thr: int, hard_task_lst: Optional[list[str]] = None) -> bool:
    sanity_checks()

    current_finished_status = [PENDING_STATUS, FINISHED_STATUS, FAILED_STATUS]

    # Get all eval request that are FINISHED, if you want to run other evals, change this parameter
    eval_requests: list[EvalRequest] = get_eval_requests(job_status=current_finished_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)
    # Sort the evals by priority (first submitted, first run)
    eval_requests: list[EvalRequest] = sort_models_by_priority(api=API, models=eval_requests)

    random.shuffle(eval_requests)

    eval_results: list[EvalResult] = get_raw_eval_results(EVAL_RESULTS_PATH_BACKEND, EVAL_REQUESTS_PATH_BACKEND)

    result_name_to_request = {request_to_result_name(r): r for r in eval_requests}
    result_name_to_result = {r.eval_name: r for r in eval_results}

    for eval_request in eval_requests:
        if eval_request.likes >= thr:
            result_name: str = request_to_result_name(eval_request)

            # Check the corresponding result
            eval_result: Optional[EvalResult] = result_name_to_result[result_name] if result_name in result_name_to_result else None

            task_lst = TASKS_HARNESS.copy()
            random.shuffle(task_lst)
            # TODO: Configure individual task parameters, don't need to update because only
            # `run_standard_mode` goes through this path.

            # Iterate over tasks and, if we do not have results for a task, run the relevant evaluations
            for task in task_lst:
                task_name = task.benchmark

                do_run_task = False
                if hard_task_lst is None or any(ss in task_name for ss in hard_task_lst):
                    do_run_task = True

                task_lst = ['nq', 'trivia', 'tqa', 'self']
                if (eval_result is None or do_run_task or task_name not in eval_result.results or
                        any(ss in task_name for ss in task_lst)):
                    eval_request: EvalRequest = result_name_to_request[result_name]

                    my_snapshot_download(repo_id=QUEUE_REPO, revision="main",
                                         local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
                    my_set_eval_request(api=API, eval_request=eval_request, set_to_status=RUNNING_STATUS,
                                        hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)

                    results = process_evaluation(task, eval_request, sync_to_s3=False)

                    my_snapshot_download(repo_id=QUEUE_REPO, revision="main",
                                         local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
                    my_set_eval_request(api=API, eval_request=eval_request, set_to_status=FINISHED_STATUS,
                                        hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)

                    return True

    return False


def process_pending_requests() -> bool:
    # TODO: Configure individual task parameters, don't need to update because only
    # `run_standard_mode` goes through this path.
    # this path is also unused, not factoring in now.
    sanity_checks()

    current_pending_status = [PENDING_STATUS]

    # Get all eval request that are PENDING, if you want to run other evals, change this parameter
    eval_requests = get_eval_requests(job_status=current_pending_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)
    # Sort the evals by priority (first submitted, first run)
    eval_requests = sort_models_by_priority(api=API, models=eval_requests)

    random.shuffle(eval_requests)

    logging.info("Found %d %s eval requests", len(eval_requests), ','.join(current_pending_status))

    if len(eval_requests) == 0:
        return False

    eval_request = eval_requests[0]
    logging.info("Processing eval request: %s", eval_request)

    my_snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
    my_set_eval_request(api=API, eval_request=eval_request, set_to_status=RUNNING_STATUS, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)

    # TODO: Fix tasks names that are configured here.
    task_lst = TASKS_HARNESS.copy()
    random.shuffle(task_lst)

    for task in task_lst:
        results = process_evaluation(task, eval_request)

    my_snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)
    my_set_eval_request(api=API, eval_request=eval_request, set_to_status=FINISHED_STATUS, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND)

    return True


def get_model_alias(model_name):
    model_alias = model_name
    if model_name.startswith('s3://'):
        config_file = Path("outputs") / model_name.split("/")[-1] / "config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            P = config.get('P', 'unknown')
            run_id = Path(config.get('output_dir')).name

            # Update model_name in results to ParControl/P={P}/{actual_run_id}
            model_alias = f"ParControl/P={P}/{run_id}"
    return model_alias


def evaluate_single_model(
    model_name: str,
    precision: str = "float16",
    model_type: str = "pretrained",
    batch_size: str = "1",
    shuffle: bool = True,
    sample_limit: int = None,
    sync_to_s3: bool = False,
    s3_base_dir: str = None,
    force: bool = False,
    seed: int = 42,
    eval_benchmark_tasks: list[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Modal remote function: Evaluate a single HuggingFace model on all tasks."""
    logging.info("Starting remote evaluation for model: %s", model_name)
    if sample_limit:
        logging.info("Limiting each evaluation to %s examples", sample_limit)

    # Create EvalRequest for the specified model
    eval_request = EvalRequest(
        model=model_name,
        private=False,
        status="RUNNING",
        json_filepath="",  # Not needed for local evaluation
        weight_type="Original",
        model_type=model_type,
        precision=precision,
        revision="main",
        submitted_time=datetime.now().isoformat()
    )
    eval_request.load_model()  # Load model directly

    # Run evaluations on specified tasks
    task_lst = get_tasks_by_benchmarks(eval_benchmark_tasks)
    if shuffle:
        task_lst = random.sample(task_lst, len(task_lst))
    all_results = {}

    model_alias = get_model_alias(model_name)
    for task in task_lst:
        s3_target_dir = s3_base_dir
        # if model_name.startswith("s3://"):  # Note: Future enhancement to organize results by model directory
        #     s3_target_dir = model_name.rstrip("/") + "/evals"

        # Check if this specific task result already exists
        if sync_to_s3:
            s3_path = check_s3_results_exist(s3_target_dir, model_alias, task.benchmark)
            if s3_path is not None and not force:
                logging.info("Skipping task %s for model %s; result already exists in S3: %s",
                             task.benchmark, model_name, s3_path)
                continue

        try:
            results = process_evaluation(
                task,
                eval_request,
                upload_to_hub=False,
                batch_size=batch_size,
                model_alias=model_alias,
                sample_limit=sample_limit,
                sync_to_s3=sync_to_s3,
                s3_base_dir=s3_target_dir,
                seed=seed,
            )
            all_results[task.benchmark] = results
        except Exception as e:
            logging.error("Failed task %s", task.benchmark, exc_info=True)
            all_results[task.benchmark] = {"error": str(e)}

    # Clean up any temporary resources
    eval_request.cleanup()

    output_path = os.path.join(EVAL_RESULTS_PATH_BACKEND, *model_alias.split("/"),
                               f"all_results_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: '<not serializable>')

    # Upload aggregated results to S3 if requested
    if sync_to_s3:
        upload_to_s3(output_path, s3_base_dir + f"/{model_alias}", None, logging.getLogger(__name__))

    logging.info("All results saved to: %s", output_path)
    return all_results


def run_standard_mode():
    """Standard mode: Process evaluation queue."""
    initialize_repositories()

    wait = True
    hard_task_lst = None
    if socket.gethostname() in {'hamburg', 'neuromancer'} or os.path.isdir("/home/pminervi"):
        wait = False
        hard_task_lst = None  # ['nq', 'trivia', 'tqa']

    if wait:
        time.sleep(60 * random.randint(2, 5))

    res = False

    if res is False:
        if random.randint(0, 5) == 0:
            res = maybe_refresh_results(0, hard_task_lst=hard_task_lst)
        else:
            res = process_finished_requests(0, hard_task_lst=hard_task_lst)


@app.function(image=MODAL_IMAGE,
              gpu=MODAL_GPU,
              timeout=3600 * 24,  # 2 hour timeout for long evaluations
              volumes={
                  "/root/outputs": modal.Volume.from_name("parcontrol-data", create_if_missing=True),
              },
              secrets=[
                  modal.Secret.from_name("aws"),
              ])
def evaluate_all_models(model_names, use_cache=False, **kwargs):
    now = pd.Timestamp.now(tz="US/Pacific")
    log_path = Path("outputs") / now.strftime("eval.%Y-%m-%d-%H-%M-%S.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if use_cache:
        os.environ["HF_HOME"] = "/root/outputs/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/root/outputs/transformers"
        for cache_dir in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"]]:
            os.makedirs(cache_dir, exist_ok=True)

    for model_name in model_names:
        logging.info("Starting evaluation for %s", model_name)
        # Pass all kwargs, including eval_benchmark_tasks, to evaluate_single_model
        evaluate_single_model(model_name=model_name, **kwargs)
    upload_to_s3(str(log_path), kwargs["s3_base_dir"], None, logging.getLogger(__name__))


def parse_args(argv=sys.argv):
    now = pd.Timestamp.now(tz="US/Pacific")
    parser = argparse.ArgumentParser(description="Backend CLI for Hallucination Leaderboard")
    parser.add_argument("--precision", type=str, default="float32", choices=["float16", "float32", "bfloat16"],
                        help="Model precision for local mode (default: float32)")
    parser.add_argument("--model-type", type=str, default="pretrained",
                        help="Model type for local mode (default: pretrained)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation (use 'auto' for automatic detection)")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Limit number of examples per evaluation (default: None for full dataset)")
    parser.add_argument("--no-sync-to-s3", dest="sync_to_s3", action="store_false",
                        help="Upload results to S3 (requires --s3-base-dir)")
    parser.add_argument("--s3-base-dir", type=str, default="s3://obviouslywrong-parcontrol/nd-lora/evals",
                        help="S3 base directory for uploading results")
    parser.add_argument("--force", action="store_true", help="Build even copy exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--eval-benchmark-tasks", type=str, nargs="*", default=None,
                        help="List of benchmark names to evaluate (e.g., 'nq8' 'truthfulqa_mc1'). If not specified, all tasks are evaluated.")
    parser.add_argument("model_names", type=str, nargs="+",
                        help="HuggingFace model name to evaluate in local mode (e.g., 'microsoft/DialoGPT-medium')")
    args = parser.parse_args(argv)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.model_names is not None:
        kwargs = vars(args)
        model_names = kwargs.pop("model_names")
        for name in model_names:
            evaluate_single_model(model_name=name, **kwargs)

    else:
        # No need to handle subsets of evals, this path isn't triggered from eval_experiments.py
        run_standard_mode()
