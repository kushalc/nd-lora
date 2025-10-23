"""
Paper-essential model checkpoints for ParControl reproduction.
Contains only the checkpoints referenced in the paper:
"Neural Diversity Regularizes Hallucinations in Small Language Models"

All models based on Qwen2.5-0.5B, trained on 20M tokens from The Pile.
"""

# Core paper checkpoints (Tables 1, 7, 8, 9)
CORE_CHECKPOINTS = {
    # P=1 baselines (parameter-matched)
    "Qwen2.5-0.5B_P1_R32": "s3://obviouslywrong-parcontrol/nd-lora/Qwen2.5-0.5B_P1_R32",
    "Qwen2.5-0.5B_P1_R64": "s3://obviouslywrong-parcontrol/nd-lora/Qwen2.5-0.5B_P1_R64",
    "Qwen2.5-0.5B_P1_R128": "s3://obviouslywrong-parcontrol/nd-lora/Qwen2.5-0.5B_P1_R128",

    # ParScale baselines (no LoRA, no Barlow Twins)
    "ParScale_P2_R32": "s3://obviouslywrong-parcontrol/nd-lora/ParScale_P2_R32",
    "ParScale_P4_R64": "s3://obviouslywrong-parcontrol/nd-lora/ParScale_P4_R64",
    "ParScale_P8_R128": "s3://obviouslywrong-parcontrol/nd-lora/ParScale_P8_R128",

    # ND-LoRA main results (optimized with Optuna)
    "ND-LoRA_P2": "s3://obviouslywrong-parcontrol/nd-lora/ND-LoRA_P2",
    "ND-LoRA_P4": "s3://obviouslywrong-parcontrol/nd-lora/ND-LoRA_P4",
    "ND-LoRA_P8": "s3://obviouslywrong-parcontrol/nd-lora/ND-LoRA_P8",
}

# Ablation checkpoints (Table 4)
ABLATION_CHECKPOINTS = {
    "ParScale-BT_P4": "s3://obviouslywrong-parcontrol/nd-lora/ParScale-BT_P4",
    "Stream_LoRA_P4": "s3://obviouslywrong-parcontrol/nd-lora/Stream_LoRA_P4",
    "Stream_LoRA-BT_P4": "s3://obviouslywrong-parcontrol/nd-lora/Stream_LoRA-BT_P4",
    "ND-LoRA_P4_Original": "s3://obviouslywrong-parcontrol/nd-lora/ND-LoRA_P4_Original",
}

# Module ablations (Table 6)
MODULE_ABLATION_CHECKPOINTS = {
    "ND-LoRA_P4_no_MLP": "s3://obviouslywrong-parcontrol/nd-lora/ND-LoRA_P4_no_MLP",
    "ND-LoRA_P4_no_attention": "s3://obviouslywrong-parcontrol/nd-lora/ND-LoRA_P4_no_attention",
}

# Combined dictionary of all paper-essential checkpoints
ALL_CHECKPOINTS = {
    **CORE_CHECKPOINTS,
    **ABLATION_CHECKPOINTS,
    **MODULE_ABLATION_CHECKPOINTS,
}

# Human-readable names for display
MODEL_NAMES = {
    # Core
    "Qwen2.5-0.5B_P1_R32": "Qwen2.5-0.5B (P=1, R=32)",
    "Qwen2.5-0.5B_P1_R64": "Qwen2.5-0.5B (P=1, R=64)",
    "Qwen2.5-0.5B_P1_R128": "Qwen2.5-0.5B (P=1, R=128)",
    "ParScale_P2_R32": "ParScale (P=2, R=32)",
    "ParScale_P4_R64": "ParScale (P=4, R=64)",
    "ParScale_P8_R128": "ParScale (P=8, R=128)",
    "ND-LoRA_P2": "ND-LoRA (P=2, OptC9)",
    "ND-LoRA_P4": "ND-LoRA (P=4, OptC9)",
    "ND-LoRA_P8": "ND-LoRA (P=8, OptC9)",
    # Ablations
    "ParScale-BT_P4": "ParScale-BT (P=4)",
    "Stream_LoRA_P4": "Stream-LoRA (P=4)",
    "Stream_LoRA-BT_P4": "Stream-LoRA-BT (P=4)",
    "ND-LoRA_P4_Original": "ND-LoRA Original HP (P=4)",
    # Module Ablations
    "ND-LoRA_P4_no_MLP": "ND-LoRA w/o MLP (P=4)",
    "ND-LoRA_P4_no_attention": "ND-LoRA w/o Attention (P=4)",
}

MODEL_SPACERS = [
    "ParControl Q0.5B P=2: Repro LoRA R32",
    "ParControl Q0.5B P=4: Repro LoRA R64",
    "ParControl Q0.5B P=8: Repro LoRA R128",
]

# Base model checkpoints
BASE_CHECKPOINTS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]
