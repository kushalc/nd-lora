#!/usr/bin/env python3
"""Ablation score analysis for Table 3 - analyzes performance of different variants from parquet data."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_ablation_scores(df, variants, metrics):
    """Analyze performance scores for ablation variants."""

    # Focus on Q0.5B models only
    df_q05 = df[df.index.get_level_values(0) == "Q0.5B"]

    results = {}
    baseline_score = None
    for variant_name, (p_val, lora_config) in variants.items():
        # Filter for this variant
        df_variant = df_q05[
            (df_q05.index.get_level_values(1) == p_val) &
            (df_q05.index.get_level_values(2) == lora_config)
        ]

        # Get first row and extract metric values, fail if any are missing
        try:
            assert not df_variant.empty, f"No data found for {p_val} {lora_config}"

            # Calculate average score across hallucination metrics using vectorized operations
            # Verify all metrics exist
            missing_metrics = set(halluc_metrics) - set(df_variant.columns)
            assert not missing_metrics, f"Metrics not found in columns: {missing_metrics}"

            metric_values = df_variant[metrics].iloc[0]
            assert metric_values.notna().all(), f"Missing values for variant {variant_name}: {metric_values[metric_values.isna()].index.tolist()}"

            avg_score = metric_values.mean()
            results[variant_name] = avg_score

            # Set baseline for Standard variant
            if variant_name == "Standard":
                baseline_score = avg_score
        except:
            logging.warning("Couldn't process %s: %s %s", variant_name, p_val, lora_config, exc_info=True)

    assert baseline_score is not None and baseline_score > 0, "Invalid baseline score"

    improvements = {}
    for variant_name, score in results.items():
        improvement = ((score - baseline_score) / baseline_score) * 100
        improvements[variant_name] = improvement
        logger.info("%s: %+5.1f%% (score: %.4f)", variant_name.ljust(15), improvement, score)

    return results, improvements


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(funcName)s %(message)s")

    # Aggregated score parquet is regenerated on demand from raw S3 evals.
    # NOTE: the index model levels (e.g. 'Repro LoRA R64', 'nOSL IndLoRA') come from
    # analyze_experiments.parse_model_metadata() applied to the clean eval-dir names —
    # verify these against a freshly built parquet.
    from build_scores import ensure_scores_parquet
    df = pd.read_parquet(ensure_scores_parquet(plot_type="pub", mode="deep"))

    # Define ablation variants and their mapping to data
    variants = {
        "Standard": ("P=1", "Repro LoRA R64"),          # Best P=1 baseline
        "ParScale": ("P=4", "SharedLoRA R64"),          # Shared LoRA, no BT
        "ParScale-BT": ("P=4", "nOSL SharedLoRA R64"),  # Shared LoRA + BT  # FIXME: Replace with OptC9 variant?
        "Stream LoRA": ("P=4", "IndLoRA"),              # Independent LoRA, no BT
        "Stream LoRA-BT": ("P=4", "nOSL IndLoRA"),      # Independent LoRA + BT
        "ND-LoRA": ("P=4", "ND-LoRA [OptC9]")           # Independent LoRA + BT
    }

    # Hallucination metrics to average
    halluc_metrics = df.columns
    halluc_metrics = [
        "HaluEval Dialog, Accuracy", "HaluEval QA, Accuracy", "HaluEval Summarization, Accuracy",
        "MemoTrap, Accuracy", "TruthfulQA MC1, Accuracy", "TruthfulQA MC2, Accuracy"
    ]
    scores, improvements = analyze_ablation_scores(df, variants, halluc_metrics)
