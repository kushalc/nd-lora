# NOTE: the S3 URLs below are used only as JOIN KEYS — they match the model identifier
# stored *inside* each eval artifact (pkl `model` field / results JSON config.model_name),
# which was written at eval time and is unaffected by the bucket migration / clean-name rename.
# They are not used for I/O. For loadable checkpoint paths see utils/model_checkpoints_paper.py.
MODEL_CHECKPOINTS = {
    # "P2": "s3://obviouslywrong-parcontrol/ParControl/2025-09-14-15-20-01",
    # "P4": "s3://obviouslywrong-parcontrol/ParControl/2025-09-03-00-41-55",
    # "cP2": "s3://obviouslywrong-parcontrol/ParControl/2025-09-08-13-07-09",
    # "lP2": "s3://obviouslywrong-parcontrol/ParControl/2025-09-14-15-19-49",

    # ==
    # Q0.5B
    "Qwen2.5-0.5B R32 LoRA": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-04-38",
    "Qwen2.5-0.5B R64 LoRA": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-10-20",
    "Qwen2.5-0.5B R128 LoRA": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-12-26",

    "ParScale R32 [P=2]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-14-23-00-29",
    "ParScale R64 [P=4]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-11-24-40",
    "ParScale R128 [P=8]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-14-30-12",
    "ParScale-BT R64 [P=4]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-20-48",
    # "ParScale-BT R64 [P=4, OptC9]": "s3://obviouslywrong-parcontrol/ParControl/2025-10-08-08-59-19",
    "Stream LoRA [P=4]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-24-16-57-26",

    # "slP2": "s3://obviouslywrong-parcontrol/ParControl/2025-09-14-15-19-32",
    # "slP4": "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-10-57-41",
    # "slP8": "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-14-29-49",

    # "Stream LoRA-BT [P=2]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-18-00-07",
    "Stream LoRA-BT [P=4]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-18-44-51",
    # "Stream LoRA-BT [P=8]": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-59-00",

    # Hyperparameter optimization
    # "ND-LoRA [P=2, OptB8]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP2-2025-10-06-02-23-18-trial-008",
    # "ND-LoRA [P=4, OptB8]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP2-2025-10-06-10-23-21-trial-000",
    # "ND-LoRA [P=8, OptB8]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP2-2025-10-06-10-23-36-trial-000",
    "ND-LoRA [P=2]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP-2025-10-06-12-37-36-trial-009",
    "ND-LoRA [P=4]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP-2025-10-07-03-12-51-trial-001",
    "ND-LoRA [P=8]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP-2025-10-07-03-12-00-trial-002",

    # Dynamic Router Experiments
    # "ND-LoRA [P=2]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP-2025-10-06-12-37-36-trial-009",
    # "ND-LoRA [P=8]": "s3://obviouslywrong-parcontrol/ParControl/optuna/optuna-nslP-2025-10-07-03-12-00-trial-002",
    # "ND-LoRA [P=2, Target-D P8]": "s3://obviouslywrong-parcontrol/ParControl/2025-12-14-20-55-12",

    # LoRA ablations
    # "LoRA ablation R2": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-30-03",
    # "LoRA ablation R4": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-30-17",
    # "LoRA ablation R8": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-30-33",
    # "LoRA ablation R16": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-10-20",
    # "LoRA ablation R32a": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-34-01",
    # "LoRA ablation R64a": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-34-07",
    # "LoRA ablation R128a": "s3://obviouslywrong-parcontrol/ParControl/2025-11-26-00-34-27",

    # Ablation Experiments
    # "nosl_p4_early_layers": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-14-11-15",
    # "nosl_p4_no_mlp": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-55-46",
    # "nosl_p4_no_attention": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-55-36",
    # "nosl_p4_no_o_proj": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-55-26",
    # "nosl_p4_no_kvq_proj": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-55-15",

    # P1R256 Reference Runs
    # k"osl_p1r256_reference_run": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-18-42-46",

    # General Models
    # "p8_run1": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-59-00",
    # "p4_run1": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-18-44-51",
    # "p2_run1": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-18-00-07",
    # "p1_run1": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-18-42-46",
    # "p1_run2": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-10-20",
    # "p1_run3": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-04-38",

    # NOTE: No need to run RandK3 for ≤4 since they're definitionally identical.
    # "nslP8__RandK3": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-22-35-34",
    # "nslP8__RandKn1": "s3://obviouslywrong-parcontrol/ParControl/2025-09-23-13-54-08",

    # "slP2__MvOi": "s3://obviouslywrong-parcontrol/ParControl/2025-09-21-13-58-07",
    # "slP4__MvOi": "s3://obviouslywrong-parcontrol/ParControl/2025-09-21-14-25-29",
    # "slP8__MvOi": "s3://obviouslywrong-parcontrol/ParControl/OSL_stream_vs_mean_of_others_w_inverse_scaling_P=8_0.5b",

    # "slP8__CAG": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-02-44-12",
    # "slP8__RandK3": "s3://obviouslywrong-parcontrol/ParControl/2025-09-20-10-19-21",

    # "slP8__MvO": "s3://obviouslywrong-parcontrol/ParControl/OSL_stream_vs_mean_of_others_P=8_0.5b",
    # "slP8__K3": "s3://obviouslywrong-parcontrol/ParControl/2025-09-19-23-38-31",

    # ==
    # Q1.5B
    # "P1__Q15Br32": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-18-13",
    # "P1__Q15Br64": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-19-56",
    # "P1__Q15Br128": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-22-49",
    "P1__Q15B__OptC9Baseline": "s3://obviouslywrong-parcontrol/ParControl/2026-04-14-11-14-13",

    # "P2__Q15Br32": "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-10-58-33",
    # "P4__Q15Br64": "s3://obviouslywrong-parcontrol/ParControl/2025-09-16-23-32-19",

    # "slP2__Q15B__MvOi": "s3://obviouslywrong-parcontrol/ParControl/2025-09-21-14-55-46",
    # "slP4__Q15B__MvOi": "s3://obviouslywrong-parcontrol/ParControl/2025-09-21-14-55-38",
    # "nslP4__Q15B": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-00-59-19",
    "nslP4__Q15B__OptC9": "s3://obviouslywrong-parcontrol/ParControl/2026-04-14-03-39-39",

    # "slP2__Q15B__MvOinLOOa": "s3://obviouslywrong-parcontrol/ParControl/2025-09-21-23-53-23",
    # "slP4__Q15B__MvOinLOOa": "s3://obviouslywrong-parcontrol/ParControl/2025-09-22-10-52-14",

    # "slP2__Q15B": "s3://obviouslywrong-parcontrol/ParControl/2025-09-15-10-57-47",
    # "slP4__Q15B": "s3://obviouslywrong-parcontrol/ParControl/2025-09-17-12-17-09",
    # "slP4__Q15Bd1": "s3://obviouslywrong-parcontrol/ParControl/2025-09-18-06-02-39",
    # "slP4__Q15Bd19": "s3://obviouslywrong-parcontrol/ParControl/2025-09-19-00-04-02",
    # "slP4__Q15Bl01": "s3://obviouslywrong-parcontrol/ParControl/2025-09-17-23-11-36",
    # "slP4__Q15Bl02": "s3://obviouslywrong-parcontrol/ParControl/2025-09-17-23-07-01",

    # ==
    # Q3B
    # "P2__Q3Br32": "s3://obviouslywrong-parcontrol/ParControl/2025-09-16-10-30-49",
    # "slP2__Q3B": "s3://obviouslywrong-parcontrol/ParControl/2025-09-16-10-31-07",

    # Repro
    # "repro__slP4": "s3://obviouslywrong-parcontrol/ParControl/2025-09-20-15-28-02",  # FIXME: Breaks?
    # "repro__slP2__clean_cache": "s3://obviouslywrong-parcontrol/ParControl/2025-09-20-19-35-07",
    # "repro__slP2__259_clean_rewind": "s3://obviouslywrong-parcontrol/ParControl/2025-09-20-21-02-44",
}

# FIXME: Merge this with the above, but not today.
MODEL_NAMES = {
    # ==
    # Q0.5B
    "ParControl/P=1/2025-09-22-00-04-38": "ParControl Q0.5B P=1: Repro LoRA R32",  # Parameter matched to P=2, 4 & 8
    "ParControl/P=1/2025-09-22-00-10-20": "ParControl Q0.5B P=1: Repro LoRA R64",
    "ParControl/P=1/2025-09-22-00-12-26": "ParControl Q0.5B P=1: Repro LoRA R128",

    "ParControl/P=2/2025-09-14-23-00-29": "ParControl Q0.5B P=2: SharedLoRA R32",
    "ParControl/P=4/2025-09-15-11-24-40": "ParControl Q0.5B P=4: SharedLoRA R64",
    "ParControl/P=8/2025-09-15-14-30-12": "ParControl Q0.5B P=8: SharedLoRA R128",
    "ParControl/P=4/2025-09-23-13-20-48": "ParControl Q0.5B P=4: nOSL SharedLoRA R64",
    "ParControl/P=4/2025-10-08-08-59-19": "ParControl Q0.5B P=4: nOSL SharedLoRA R64 [OptC9]",
    "ParControl/P=4/2025-09-24-16-57-26": "ParControl Q0.5B P=4: IndLoRA",

    "ParControl/P=2/2025-09-22-18-00-07": "ParControl Q0.5B P=2: nOSL IndLoRA",
    "ParControl/P=4/2025-09-22-18-44-51": "ParControl Q0.5B P=4: nOSL IndLoRA",
    "ParControl/P=8/2025-09-22-00-59-00": "ParControl Q0.5B P=8: nOSL IndLoRA",
    "ParControl/P=8/2025-09-22-22-35-34": "ParControl Q0.5B P=8: nOSL IndLoRA RandK3",
    "ParControl/P=8/2025-09-23-13-54-08": "ParControl Q0.5B P=8: nOSL IndLoRA RandKn1",

    "ParControl/P=2/2025-09-21-13-58-07": "ParControl Q0.5B P=2: MvOiOSL IndLoRA",
    "ParControl/P=4/2025-09-21-14-25-29": "ParControl Q0.5B P=4: MvOiOSL IndLoRA",
    "ParControl/P=8/OSL_stream_vs_mean_of_others_w_inverse_scaling_P=8_0.5b": "ParControl Q0.5B P=8: MvOiOSL IndLoRA",

    # experiments
    "ParControl/P=2/2025-09-03-10-39-24": "ParControl Q0.5B P=2: Repro PfxAggOnly",
    "ParControl/P=2/2025-09-14-15-19-32": "ParControl Q0.5B P=2: OSL IndLoRA",
    "ParControl/P=2/2025-09-14-15-19-49": "ParControl Q0.5B P=2: IndLoRA",
    "ParControl/P=2/2025-09-14-15-20-01": "ParControl Q0.5B P=2: Repro LoRA",
    "ParControl/P=2/2025-09-20-19-35-07": "ParControl Q0.5B P=2: OSL IndLoRA [REPRO #1]",
    "ParControl/P=2/2025-09-20-21-02-44": "ParControl Q0.5B P=2: OSL IndLoRA [REPRO #2]",
    "ParControl/P=4/2025-09-03-00-41-55": "ParControl Q0.5B P=4: Repro PfxAggOnly",
    "ParControl/P=4/2025-09-15-10-57-41": "ParControl Q0.5B P=4: OSL IndLoRA",
    "ParControl/P=4/2025-09-17-13-29-42": "ParControl Q0.5B P=4: VarNormOSL IndLoRA",
    "ParControl/P=8/2025-09-15-14-29-49": "ParControl Q0.5B P=8: OSL IndLoRA",
    "ParControl/P=8/2025-09-16-23-11-36": "ParControl Q0.5B P=8: VarNormOSL IndLoRA",
    "ParControl/P=8/2025-09-19-23-38-31": "ParControl Q0.5B P=8: OSL IndLoRA K3",
    "ParControl/P=8/2025-09-20-10-19-21": "ParControl Q0.5B P=8: OSL IndLoRA RandK3",
    "ParControl/P=8/OSL_stream_vs_mean_of_others_P=8_0.5b": "ParControl Q0.5B P=8: MvOOSL IndLoRA",

    'ParControl/P=2/2025-09-22-15-45-08': "Nirmal FIXME",
    'ParControl/P=4/2025-09-22-15-38-58': "Nirmal FIXME",
    "ParControl/P=2/2025-09-21-01-40-23": "Nirmal FIXME",
    "ParControl/P=2/2025-09-21-08-49-20": "Nirmal FIXME",
    "ParControl/P=4/2025-09-22-00-59-19": "ParControl Q1.5B P=4: nOSL IndLoRA D19",
    "ParControl/P=8/2025-09-21-18-54-30": "Nirmal FIXME",
    "ParControl/P=8/2025-09-21-20-03-35": "Nirmal FIXME",
    "ParControl/P=8/2025-09-21-20-12-09": "Nirmal FIXME",
    "ParControl/P=8/2025-09-22-00-59-00": "ParControl Q0.5B P=8: nOSL IndLoRA",
    "ParControl/P=8/2025-09-22-02-10-32": "xParControl Q0.5B P=8: OSL IndLoRA TAG",
    "ParControl/P=8/2025-09-22-02-44-12": "xParControl Q0.5B P=8: OSL IndLoRA CAG",
    "ParControl/P=4/2025-09-23-13-55-15": "ParControl Q0.5B P=4: [A] nOSL IndLORA - No KVQ",
    "ParControl/P=4/2025-09-23-13-55-26": "ParControl Q0.5B P=4: [A] nOSL IndLORA - No O Proj",
    "ParControl/P=4/2025-09-23-13-55-36": "ParControl Q0.5B P=4: [A] nOSL IndLORA - No Attention",
    "ParControl/P=4/2025-09-23-14-11-35": "ParControl Q0.5B P=4: [A] nOSL IndLORA - Design Layers",
    "ParControl/P=4/2025-09-23-13-55-46": "ParControl Q0.5B P=4: [A] nOSL IndLORA - No MLP",
    'ParControl/P=4/2025-09-23-14-11-25': "ParControl Q0.5B P=4: [A] nOSL IndLORA - Mid Layers",
    'ParControl/P=4/2025-09-23-14-11-45': "ParControl Q0.5B P=4: [A] nOSL IndLORA - Late Layers",
    "ParControl/P=4/2025-09-23-14-11-15": "ParControl Q0.5B P=4: [A] nOSL IndLORA - Early Layers",

    # P=1 Remaining Experiments
    "ParControl/P=1/2025-09-23-18-42-46": "Nirmal FIXME",
    "ParControl/P=1/2025-09-25-00-08-38": "Nirmal FIXME",
    "ParControl/P=1/2025-09-25-00-09-40": "Nirmal FIXME",
    "ParControl/P=1/2025-09-25-00-10-33": "Nirmal FIXME",
    "ParControl/P=1/2025-09-25-00-11-08": "Nirmal FIXME",
    "ParControl/P=3/2025-09-25-00-05-27": "Nirmal FIXME",
    "ParControl/P=6/2025-09-25-00-06-58": "Nirmal FIXME",
    "ParControl/P=7/2025-09-25-00-07-51": "Nirmal FIXME",

    'ParControl/P=1/2025-09-25-00-08-38': "Nirmal FIXME",
    'ParControl/P=1/2025-09-25-00-09-40': "Nirmal FIXME",
    'ParControl/P=1/2025-09-25-00-10-33': "Nirmal FIXME",
    'ParControl/P=1/2025-09-25-00-11-08': "Nirmal FIXME",
    'ParControl/P=2/optuna-nslP2-2025-10-06-02-23-18-trial-008': "ParControl Q0.5B P=2: ND-LoRA [OptB8]",
    'ParControl/P=4/optuna-nslP2-2025-10-06-10-23-21-trial-000': "ParControl Q0.5B P=4: ND-LoRA [OptB8]",
    'ParControl/P=8/optuna-nslP2-2025-10-06-10-23-36-trial-000': "ParControl Q0.5B P=8: ND-LoRA [OptB8]",
    "ParControl/P=2/optuna-nslP-2025-10-06-12-37-36-trial-009": "ParControl Q0.5B P=2: ND-LoRA [OptC9]",
    "ParControl/P=4/optuna-nslP-2025-10-07-03-12-51-trial-001": "ParControl Q0.5B P=4: ND-LoRA [OptC9]",
    "ParControl/P=8/optuna-nslP-2025-10-07-03-12-00-trial-002": "ParControl Q0.5B P=8: ND-LoRA [OptC9]",
    "ParControl/P=2/2025-12-14-20-55-12": "ParControl Q0.5B P=2: ND-LoRA [OptC9, Target-D P8]",
    "ParControl/P=8-StreamSelect-2/optuna-nslP-2025-10-07-03-12-00-trial-002": "ParControl Q0.5B P=8→2: Stream Selection (No Renorm)",
    "ParControl/P=1/2025-10-11-17-41-35": "FIXME",
    "ParControl/P=1/2025-10-11-17-42-29": "FIXME",
    "ParControl/P=2/2025-10-10-15-01-35": "FIXME",
    "ParControl/P=2/2025-10-10-15-02-43": "FIXME",
    "ParControl/P=4/2025-10-10-18-36-08": "FIXME",
    "ParControl/P=4/2025-10-10-18-41-27": "FIXME",
    "ParControl/P=2/optuna-nslP-2025-10-07-14-53-56-trial-021": "ParControl Q0.5B P=2: ND-LoRA [OptD21]",
    "ParControl/P=2/optuna-nslP-2025-10-08-08-07-31-trial-008": "ParControl Q0.5B P=2: ND-LoRA [OptE8]",
    "ParControl/P=2/2025-10-15-08-52-24": "FIXME",
    "ParControl/P=2/2025-10-15-09-52-37": "FIXME",
    "ParControl/P=2/2025-10-15-09-54-40": "FIXME",
    "ParControl/P=2/2025-10-15-12-42-21": "FIXME",
    "ParControl/P=2/2025-10-16-18-01-09": "FIXME",
    "ParControl/P=2/2025-10-17-00-59-34": "FIXME",
    "ParControl/P=2/2025-10-17-17-16-40": "FIXME",
    "ParControl/P=2/2025-10-18-11-26-31": "FIXME",
    "ParControl/P=3/2025-12-09-16-08-07": "FIXME",
    "ParControl/P=4/2025-10-13-11-17-45": "FIXME",
    "ParControl/P=4/2025-10-18-21-31-50": "FIXME",
    "ParControl/P=4/2025-10-19-07-12-57": "FIXME",
    'ParControl/P=3/2025-09-25-00-05-27': "Nirmal FIXME",
    'ParControl/P=4/2025-09-23-13-55-26': "Nirmal FIXME",
    'ParControl/P=4/2025-09-23-14-11-15': "Nirmal FIXME",
    'ParControl/P=6/2025-09-25-00-06-58': "Nirmal FIXME",
    'ParControl/P=7/2025-09-25-00-07-51': "Nirmal FIXME",

    # P=1 LoRA ablations
    "ParControl/P=1/2025-11-26-00-10-20": "LoRA ablation R16",
    "ParControl/P=1/2025-11-26-00-30-03": "LoRA ablation R2",
    "ParControl/P=1/2025-11-26-00-30-17": "LoRA ablation R4",
    "ParControl/P=1/2025-11-26-00-30-33": "LoRA ablation R8",
    "ParControl/P=1/2025-11-26-00-34-01": "LoRA ablation R32a",
    "ParControl/P=1/2025-11-26-00-34-07": "LoRA ablation R64a",
    "ParControl/P=1/2025-11-26-00-34-27": "LoRA ablation R128a",

    # Q1.5B
    "ParControl/P=1/2025-09-22-00-18-13": "ParControl Q1.5B P=1: Repro LoRA R32",  # Parameter matched to P=2, 4 & 8
    "ParControl/P=1/2025-09-22-00-19-56": "ParControl Q1.5B P=1: Repro LoRA R64",
    "ParControl/P=1/2025-09-22-00-22-49": "ParControl Q1.5B P=1: Repro LoRA R128",

    "ParControl/P=2/2025-09-15-10-58-33": "ParControl Q1.5B P=2: Repro LoRA R32",
    "ParControl/P=4/2025-09-16-23-32-19": "ParControl Q1.5B P=4: Repro LoRA R64",

    "ParControl/P=2/2025-09-21-14-55-46": "ParControl Q1.5B P=2: MvOiOSL IndLoRA D19",
    "ParControl/P=4/2025-09-21-14-55-38": "ParControl Q1.5B P=4: MvOiOSL IndLoRA D19",

    # experiments
    "ParControl/P=2/2025-09-21-23-53-23": "ParControl Q1.5B P=2: MvOinOSL LOOa IndLoRA D19",
    "ParControl/P=2/2025-09-15-10-57-47": "ParControl Q1.5B P=2: OSL IndLoRA",
    "ParControl/P=4/2025-09-16-10-21-07": "ParControl Q1.5B P=4: Repro LoRA R64 [UNDERTRAINED]",
    "ParControl/P=4/2025-09-16-10-21-25": "ParControl Q1.5B P=4: OSL IndLoRA [UNDERTRAINED]",
    "ParControl/P=4/2025-09-17-12-17-09": "ParControl Q1.5B P=4: OSL IndLoRA",
    "ParControl/P=4/2025-09-17-23-11-36": "ParControl Q1.5B P=4: OSL IndLoRA L0.01",
    "ParControl/P=4/2025-09-17-23-07-01": "ParControl Q1.5B P=4: OSL IndLoRA L0.02",
    "ParControl/P=4/2025-09-18-06-02-39": "ParControl Q1.5B P=4: MvOiOSL IndLoRA D1",
    "ParControl/P=4/2025-09-19-00-04-02": "ParControl Q1.5B P=4: OSL IndLoRA D19",
    "ParControl/P=4/2025-09-22-10-52-14": "ParControl Q1.5B P=4: MvOinOSL LOOa IndLoRA D19",
    "ParControl/P=4/2026-04-14-03-39-39": "ParControl Q1.5B P=4: ND-LoRA [OptC9]",
    "ParControl/P=1/2026-04-14-11-14-13": "ParControl Q1.5B P=1: OptC9 Baseline R192",
    "ParControl/P=2/2025-09-23-01-28-15": "FIXME",
    "ParControl/P=2/2025-09-23-01-28-16": "FIXME",
    "ParControl/P=2/2025-09-23-01-28-17": "FIXME",
    "ParControl/P=2/2025-09-23-01-29-30": "FIXME",
    "ParControl/P=2/2025-09-23-01-29-31": "FIXME",
    "ParControl/P=2/optuna-nslP2-2025-10-06-02-23-18-trial-008": "Kushal FIXME",

    # Q3B
    "ParControl/P=2/2025-09-16-10-30-49": "ParControl Q3B P=2: Repro LoRA R32",
    "ParControl/P=2/2025-09-16-10-31-07": "ParControl Q3B P=2: OSL IndLoRA [UNDERTRAINED]",
}

MODEL_SPACERS = [
    "ParControl Q0.5B P=2: Repro LoRA R32",
    "ParControl Q0.5B P=4: Repro LoRA R64",
    "ParControl Q0.5B P=8: Repro LoRA R128",

    # Q1.5B
    "ParControl Q1.5B P=2: Repro LoRA R32",
    "ParControl Q1.5B P=4: Repro LoRA R64 [60M]",

    # Q3B
    "ParControl Q3B P=2: Repro LoRA R32",
]

BASE_CHECKPOINTS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]
