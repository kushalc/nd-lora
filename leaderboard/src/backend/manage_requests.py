import glob
import json
import logging
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

from src.backend.s3_model_handler import S3ModelHandler
from src.utils import my_snapshot_download

logger = logging.getLogger(__name__)


@dataclass
class EvalRequest:
    model: str
    private: bool
    status: str
    json_filepath: str
    weight_type: str = "Original"
    model_type: str = ""
    precision: str = ""
    base_model: Optional[str] = None
    revision: str = "main"
    submitted_time: Optional[str] = "2022-05-18T11:40:22.519222"
    likes: Optional[int] = 0
    params: Optional[int] = None
    license: Optional[str] = ""
    _model: Optional[object] = None
    _tokenizer: Optional[object] = None
    _s3_handler: Optional[S3ModelHandler] = None

    def load_model(self):
        """Load model and tokenizer directly."""
        if self._model is not None and self._tokenizer is not None:
            return

        if self.model.startswith('s3://'):
            # S3/ParScale models - load via S3ModelHandler
            logger.info("Loading S3/ParScale model: %s", self.model)
            if self._s3_handler is None:
                self._s3_handler = S3ModelHandler()
            self._model, self._tokenizer = self._s3_handler.load_model(self.model)
            logger.info("Successfully loaded S3 model: %s", self.model)
        else:
            # HuggingFace models - don't load, let evaluator handle directly
            self._model = None
            self._tokenizer = None
            logger.info("Using direct HF evaluation for: %s", self.model)

    def get_model(self):
        """Get loaded model and tokenizer."""
        return self._model, self._tokenizer

    def cleanup(self):
        """Clean up resources."""
        if self._s3_handler is not None:
            self._s3_handler.cleanup()
            self._s3_handler = None
        self._model = None
        self._tokenizer = None


def set_eval_request(api: HfApi, eval_request: EvalRequest, set_to_status: str, hf_repo: str, local_dir: str):
    """Updates a given eval request with its new status on the hub."""
    json_filepath = eval_request.json_filepath

    with open(json_filepath) as fp:
        data = json.load(fp)

    data["status"] = set_to_status

    with open(json_filepath, "w") as f:
        f.write(json.dumps(data))

    api.upload_file(path_or_fileobj=json_filepath, path_in_repo=json_filepath.replace(local_dir, ""),
                    repo_id=hf_repo, repo_type="dataset")


def get_eval_requests(job_status: list, local_dir: str, hf_repo: str) -> list[EvalRequest]:
    """Get all pending evaluation requests."""
    my_snapshot_download(repo_id=hf_repo, revision="main", local_dir=local_dir, repo_type="dataset", max_workers=60)

    eval_requests = []
    json_files = glob.glob(f"{local_dir}/**/*.json", recursive=True)

    for json_filepath in json_files:
        with open(json_filepath) as fp:
            data = json.load(fp)

        if data["status"] in job_status:
            data["json_filepath"] = json_filepath
            eval_request = EvalRequest(**data)
            eval_requests.append(eval_request)

    return eval_requests


def check_completed_evals(api: HfApi, checked_status: str, completed_status: str, failed_status: str,
                          hf_repo: str, local_dir: str, hf_repo_results: str, local_dir_results: str):
    """Check completed evaluations and update their status."""
    my_snapshot_download(repo_id=hf_repo_results, revision="main", local_dir=local_dir_results,
                         repo_type="dataset", max_workers=60)

    eval_requests = get_eval_requests(job_status=[checked_status], local_dir=local_dir, hf_repo=hf_repo)

    for eval_request in eval_requests:
        model_result_filepaths = glob.glob(f"{local_dir_results}/{eval_request.model}/results_*.json")

        if len(model_result_filepaths) > 0:
            eval_request.status = completed_status
            set_eval_request(api, eval_request, completed_status, hf_repo, local_dir)
