import os
from dataclasses import dataclass
from enum import Enum

import torch

from src.envs import CACHE_PATH


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str
    num_fewshot: int


class Tasks(Enum):
    task11 = Task("nq8", "em", "NQ Open 8", 8)
    task12 = Task("tqa8", "em", "TriviaQA 8", 8)

    task21 = Task("popqa", "em", "PopQA", 8)

    # TruthfulQA is intended as a zero-shot benchmark [5, 47]. https://owainevans.github.io/pdfs/truthfulQA_lin_evans.pdf
    task3 = Task("truthfulqa_mc1", "acc", "TruthfulQA MC1", 0)
    task4 = Task("truthfulqa_mc2", "acc", "TruthfulQA MC2", 0)

    task5 = Task("halueval_qa", "acc", "HaluEval QA", 0)
    task6 = Task("halueval_dialogue", "acc", "HaluEval Dialogue", 0)
    task7 = Task("halueval_summarization", "acc", "HaluEval Summarization", 0)

    # Note: These tasks are too slow and provide insufficient signal
    # task0 = Task("nq_open", "em", "NQ Open", 64)  # 64, as in the ATLAS paper
    # task1 = Task("triviaqa", "em", "TriviaQA", 64)  # 64, as in the ATLAS paper
    # task2 = Task("truthfulqa_gen", "rougeL_acc", "TruthfulQA Gen", 0)
    # task8 = Task("xsum", "rougeL", "XSum", 2)
    # task9 = Task("cnndm", "rougeL", "CNN/DM", 2)
    # task8_1 = Task("xsum_v2", "rougeL", "XSum", 0)
    # task9_1 = Task("cnndm_v2", "rougeL", "CNN/DM", 0)
    # task13 = Task("ifeval", "prompt_level_strict_acc", "IFEval", 0)

    task10 = Task("memo-trap", "acc", "memo-trap", 0)
    task10_2 = Task("memo-trap_v2", "acc", "memo-trap", 0)

    # Note: These tasks are currently broken
    # task14 = Task("selfcheckgpt", "max-selfcheckgpt", "SelfCheckGPT", 0)
    # task18 = Task("faithdial_hallu", "acc", "FaithDial", 8)
    # task19 = Task("faithdial_hallu_v2", "acc", "FaithDial", 8)

    # task15 = Task("fever10", "acc", "FEVER", 16)
    task15_1 = Task("fever11", "acc", "FEVER", 8)

    task16 = Task("squadv2", "exact", "SQuADv2", 4)
    task22 = Task("nq_swap", "em", "NQ-Swap", 4)

    task17 = Task("truefalse_cieacf", "acc", "TrueFalse", 8)

    task20 = Task("race", "acc", "RACE", 0)
    task23 = Task("race_4", "acc", "RACE", 0)

    # General purpose evaluations
    task24 = Task("wikitext", "word_perplexity", "WikiText", 0)
    task25 = Task("pile", "word_perplexity", "Pile", 0)
    task26 = Task("winogrande", "acc", "WinoGrande", 0)


EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_REQUESTS_PATH_BACKEND_SYNC = os.path.join(CACHE_PATH, "eval-queue-bk-sync")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

LIMIT = None  # Testing; needs to be None
