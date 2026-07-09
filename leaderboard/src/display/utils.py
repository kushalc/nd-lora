from dataclasses import dataclass, make_dataclass, field
from enum import Enum

import pandas as pd


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str


class Tasks(Enum):
    # XXX include me back at some point
    nqopen = Task("nq8", "em", "NQ Open/EM")
    triviaqa = Task("tqa8", "em", "TriviaQA/EM")

    truthfulqa_mc1 = Task("truthfulqa_mc1", "acc", "TruthQA MC1/Acc")
    truthfulqa_mc2 = Task("truthfulqa_mc2", "acc", "TruthQA MC2/Acc")
    truthfulqa_gen = Task("truthfulqa_gen", "rougeL_acc", "TruthQA Gen/ROUGE")

    xsum_r = Task("xsum_v2", "rougeL", "XSum/ROUGE")
    xsum_f = Task("xsum_v2", "factKB", "XSum/factKB")
    xsum_b = Task("xsum_v2", "bertscore_precision", "XSum/BERT-P")

    cnndm_r = Task("cnndm_v2", "rougeL", "CNN-DM/ROUGE")
    cnndm_f = Task("cnndm_v2", "factKB", "CNN-DM/factKB")
    cnndm_b = Task("cnndm_v2", "bertscore_precision", "CNN-DM/BERT-P")

    race = Task("race", "acc", "RACE/Acc")
    squadv2 = Task("squadv2", "exact", "SQuADv2/EM")

    memotrap = Task("memo-trap_v2", "acc", "MemoTrap/Acc")
    ifeval = Task("ifeval", "prompt_level_strict_acc", "IFEval/Acc")

    faithdial = Task("faithdial_hallu_v2", "acc", "FaithDial/Acc")

    halueval_qa = Task("halueval_qa", "acc", "HaluQA/Acc")
    halueval_summ = Task("halueval_summarization", "acc", "HaluSumm/Acc")
    halueval_dial = Task("halueval_dialogue", "acc", "HaluDial/Acc")

    # XXX include me back at some point
    #selfcheck = Task("selfcheckgpt", "max-selfcheckgpt", "SelfCheckGPT")

    # add popqa, fever, nq-swap and truefalse
    fever11 = Task("fever11", "acc", "FEVER/Acc")
    truefalse_cieacf = Task("truefalse_cieacf", "acc", "TrueFalse/Acc")
    popqa = Task("popqa", "em", "PopQA/EM")
    nq_swap = Task("nq_swap", "em", "NQ-Swap/EM")


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False
    dummy: bool = False

auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(["model_type_symbol", ColumnContent, field(default_factory=lambda: ColumnContent("T", "str", True, never_hidden=True))])
auto_eval_column_dict.append(["model", ColumnContent, field(default_factory=lambda: ColumnContent("Model", "markdown", True, never_hidden=True))])

#Scores
# auto_eval_column_dict.append(["average", ColumnContent, field(default_factory=lambda: ColumnContent("Avg", "number", True))])

auto_eval_column_dict.append(["Faithfulness", ColumnContent, field(default_factory=lambda: ColumnContent("Faithfulness", "number", True))])
auto_eval_column_dict.append(["Factuality", ColumnContent, field(default_factory=lambda: ColumnContent("Factuality", "number", True))])

for task in Tasks:
    auto_eval_column_dict.append([task.name, ColumnContent, field(default_factory=lambda t=task: ColumnContent(t.value.col_name, "number", True))])

# Model information
auto_eval_column_dict.append(["model_type", ColumnContent, field(default_factory=lambda: ColumnContent("Type", "str", False))])
auto_eval_column_dict.append(["architecture", ColumnContent, field(default_factory=lambda: ColumnContent("Architecture", "str", False))])
auto_eval_column_dict.append(["weight_type", ColumnContent, field(default_factory=lambda: ColumnContent("Weight type", "str", False, True))])
auto_eval_column_dict.append(["precision", ColumnContent, field(default_factory=lambda: ColumnContent("Precision", "str", False))])
auto_eval_column_dict.append(["license", ColumnContent, field(default_factory=lambda: ColumnContent("Hub License", "str", False))])
auto_eval_column_dict.append(["params", ColumnContent, field(default_factory=lambda: ColumnContent("#Params (B)", "number", False))])
auto_eval_column_dict.append(["likes", ColumnContent, field(default_factory=lambda: ColumnContent("Hub ❤️", "number", False))])
auto_eval_column_dict.append(["still_on_hub", ColumnContent, field(default_factory=lambda: ColumnContent("Available on the hub", "bool", False))])
auto_eval_column_dict.append(["revision", ColumnContent, field(default_factory=lambda: ColumnContent("Model sha", "str", False, False))])
# Dummy column for the search bar (hidden by the custom CSS)
auto_eval_column_dict.append(["dummy", ColumnContent, field(default_factory=lambda: ColumnContent("model_name_for_query", "str", False, dummy=True))])

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    private = ColumnContent("private", "bool", True)
    precision = ColumnContent("precision", "str", True)
    weight_type = ColumnContent("weight_type", "str", "Original")
    status = ColumnContent("status", "str", True)


@dataclass
class ModelDetails:
    name: str
    symbol: str = ""  # emoji, only for the model type


class ModelType(Enum):
    PT = ModelDetails(name="pretrained", symbol="🟢")
    FT = ModelDetails(name="fine-tuned on domain-specific datasets", symbol="🔶")
    chat = ModelDetails(name="chat models (RLHF, DPO, IFT, ...)", symbol="💬")
    merges = ModelDetails(name="base merges and moerges", symbol="🤝")
    Unknown = ModelDetails(name="", symbol="?")

    def to_str(self, separator=" "):
        return f"{self.value.symbol}{separator}{self.value.name}"

    @staticmethod
    def from_str(type):
        if "fine-tuned" in type or "🔶" in type:
            return ModelType.FT
        if "pretrained" in type or "🟢" in type:
            return ModelType.PT
        if any([k in type for k in ["instruction-tuned", "RL-tuned", "chat", "🟦", "⭕", "💬"]]):
            return ModelType.chat
        if "merge" in type or "🤝" in type:
            return ModelType.merges
        return ModelType.Unknown


class WeightType(Enum):
    Adapter = ModelDetails("Adapter")
    Original = ModelDetails("Original")
    Delta = ModelDetails("Delta")


class Precision(Enum):
    float32 = ModelDetails("float32")
    float16 = ModelDetails("float16")
    bfloat16 = ModelDetails("bfloat16")
    qt_8bit = ModelDetails("8bit")
    qt_4bit = ModelDetails("4bit")
    qt_GPTQ = ModelDetails("GPTQ")
    Unknown = ModelDetails("?")

    @staticmethod
    def from_str(precision: str):
        if precision in ["torch.float32", "float32"]:
            return Precision.float32
        if precision in ["torch.float16", "float16"]:
            return Precision.float16
        if precision in ["torch.bfloat16", "bfloat16"]:
            return Precision.bfloat16
        if precision in ["8bit"]:
            return Precision.qt_8bit
        if precision in ["4bit"]:
            return Precision.qt_4bit
        if precision in ["GPTQ", "None"]:
            return Precision.qt_GPTQ
        return Precision.Unknown


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
TYPES = [c.type for c in fields(AutoEvalColumn) if not c.hidden]
COLS_LITE = [c.name for c in fields(AutoEvalColumn) if c.displayed_by_default and not c.hidden]
TYPES_LITE = [c.type for c in fields(AutoEvalColumn) if c.displayed_by_default and not c.hidden]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Tasks]

NUMERIC_INTERVALS = {
    "?": pd.Interval(-1, 0, closed="right"),
    "~1.5": pd.Interval(0, 2, closed="right"),
    "~3": pd.Interval(2, 4, closed="right"),
    "~7": pd.Interval(4, 9, closed="right"),
    "~13": pd.Interval(9, 20, closed="right"),
    "~35": pd.Interval(20, 45, closed="right"),
    "~60": pd.Interval(45, 70, closed="right"),
    "70+": pd.Interval(70, 10000, closed="right"),
}
