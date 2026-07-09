"""Single source of truth for the paper-reproduction model checkpoints.

Everything is keyed by the CLEAN eval-dir / checkpoint name used throughout the
ndlora bucket (evals/…/P=k/<clean>/ and checkpoints/<clean>/). The analysis scripts
(analyze_experiments, figure1, table5_dspec) identify a model by its clean eval-dir /
file name rather than the legacy run-id stored inside each artifact's config.model_name.

Contents are limited to exactly the checkpoints needed to reproduce the paper's
figures and tables. Base model = Qwen2.5-0.5B, 20M tokens of The Pile.
"""

# ── Storage / experiment-tracking roots (single source of truth) ─────────────────
# The bucket root and the Modal/W&B experiment tag are imported by the training,
# eval, analysis, and figure scripts so the location lives in exactly one place.
S3_BUCKET = "s3://obviouslywrong-ndlora"
MODAL_APP = "ParControl"       # Modal application / experiment codename
WANDB_PROJECT = "ParControl"   # W&B project (same experiment tag)

# ── Loadable checkpoints: clean name -> S3 path (used to RUN evals) ──────────────
# Every checkpoint lives at {S3_BUCKET}/checkpoints/<clean name>, so the dict is
# generated from the clean names to keep the bucket single-sourced.
_CHECKPOINT_NAMES = [
    "Qwen2.5-0.5B_P1_R32", "Qwen2.5-0.5B_P1_R64", "Qwen2.5-0.5B_P1_R128",
    "ParScale_P2_R32", "ParScale_P4_R64", "ParScale_P8_R128",
    "ND-LoRA_P2", "ND-LoRA_P4", "ND-LoRA_P8",
    "ParScale-BT_P4", "Stream_LoRA_P4", "Stream_LoRA-BT_P4",
    "ND-LoRA_P4_Original", "ND-LoRA_P4_no_MLP", "ND-LoRA_P4_no_attention",
]
CHECKPOINTS = {name: f"{S3_BUCKET}/checkpoints/{name}" for name in _CHECKPOINT_NAMES}
ALL_CHECKPOINTS = CHECKPOINTS  # back-compat alias

# ── Human-readable display names: clean name -> pretty label ─────────────────────
DISPLAY_NAMES = {
    "Qwen2.5-0.5B_P1_R32": "Qwen2.5-0.5B (P=1, R=32)",
    "Qwen2.5-0.5B_P1_R64": "Qwen2.5-0.5B (P=1, R=64)",
    "Qwen2.5-0.5B_P1_R128": "Qwen2.5-0.5B (P=1, R=128)",
    "ParScale_P2_R32": "ParScale (P=2, R=32)",
    "ParScale_P4_R64": "ParScale (P=4, R=64)",
    "ParScale_P8_R128": "ParScale (P=8, R=128)",
    "ND-LoRA_P2": "ND-LoRA (P=2, OptC9)",
    "ND-LoRA_P4": "ND-LoRA (P=4, OptC9)",
    "ND-LoRA_P8": "ND-LoRA (P=8, OptC9)",
    "ParScale-BT_P4": "ParScale-BT (P=4)",
    "Stream_LoRA_P4": "Stream-LoRA (P=4)",
    "Stream_LoRA-BT_P4": "Stream-LoRA-BT (P=4)",
    "ND-LoRA_P4_Original": "ND-LoRA Original HP (P=4)",
    "ND-LoRA_P4_no_MLP": "ND-LoRA w/o MLP (P=4)",
    "ND-LoRA_P4_no_attention": "ND-LoRA w/o Attention (P=4)",
}

# ── Analysis treatment names: eval-dir name -> "ParControl Q0.5B P=k: <Treatment>" ─
# analyze_experiments maps each parsed model to this string, then parse_model_metadata
# splits off (base, P, treatment); table2 / table789 / table5_score / figure1 filter on
# the treatment. Keyed by the eval-DIR name: paper models use clean names; the LoRA-rank
# sweep (Table 10) kept its run-ids, so those keys are run-ids.
MODEL_NAMES = {
    # Paper core (clean dir names)
    "Qwen2.5-0.5B_P1_R32": "ParControl Q0.5B P=1: Repro LoRA R32",
    "Qwen2.5-0.5B_P1_R64": "ParControl Q0.5B P=1: Repro LoRA R64",
    "Qwen2.5-0.5B_P1_R128": "ParControl Q0.5B P=1: Repro LoRA R128",
    "ParScale_P2_R32": "ParControl Q0.5B P=2: SharedLoRA R32",
    "ParScale_P4_R64": "ParControl Q0.5B P=4: SharedLoRA R64",
    "ParScale_P8_R128": "ParControl Q0.5B P=8: SharedLoRA R128",
    "ND-LoRA_P2": "ParControl Q0.5B P=2: ND-LoRA [OptC9]",
    "ND-LoRA_P4": "ParControl Q0.5B P=4: ND-LoRA [OptC9]",
    "ND-LoRA_P8": "ParControl Q0.5B P=8: ND-LoRA [OptC9]",
    # Table 4 architectural ablations (only present in neurodiversity evals; kept for completeness)
    "ParScale-BT_P4": "ParControl Q0.5B P=4: nOSL SharedLoRA R64",
    "Stream_LoRA_P4": "ParControl Q0.5B P=4: IndLoRA",
    "Stream_LoRA-BT_P4": "ParControl Q0.5B P=4: nOSL IndLoRA",
    # Table 10 LoRA-rank sweep (dirs kept run-ids)
    "2025-11-26-00-30-03": "ParControl Q0.5B P=1: LoRA ablation R2",
    "2025-11-26-00-30-17": "ParControl Q0.5B P=1: LoRA ablation R4",
    "2025-11-26-00-30-33": "ParControl Q0.5B P=1: LoRA ablation R8",
    "2025-11-26-00-10-20": "ParControl Q0.5B P=1: LoRA ablation R16",
    "2025-11-26-00-34-01": "ParControl Q0.5B P=1: LoRA ablation R32a",
    "2025-11-26-00-34-07": "ParControl Q0.5B P=1: LoRA ablation R64a",
    "2025-11-26-00-34-27": "ParControl Q0.5B P=1: LoRA ablation R128a",
}

MODEL_SPACERS = [
    "ParControl Q0.5B P=2: Repro LoRA R32",
    "ParControl Q0.5B P=4: Repro LoRA R64",
    "ParControl Q0.5B P=8: Repro LoRA R128",
]

# Base model checkpoints (HF hub ids)
BASE_CHECKPOINTS = [
    "Qwen/Qwen2.5-0.5B",
]
