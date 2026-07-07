#!/usr/bin/env python3
"""
Generate Figure 1: Optimal Neural Diversity

Shows U-shaped relationship between neural diversity (P) and performance across reliability tasks.
Overlays empirical bootstrap curves with Theorem 2's theoretical prediction, which models
correlation growth ρ(P) = ρ₀ + β(P-1)^γ to explain non-monotonic scaling behavior.

The x-axis shows ΔP (distance from optimal P) and y-axis shows relative performance
normalized by P=1 baseline.

Repro (R^2=0.943):
python adhoc/table4_dspec_ablations.py --since=24h
python adhoc/figure1_optimal_diversity.py
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.optimize import minimize, minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils.model_checkpoints import MODEL_NAMES

# Suppress numerical warnings from LOWESS fitting
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "paper/assets"
EVALS_DIR = BASE_DIR / "outputs/evals-full/ParControl"
CACHE_DIR = BASE_DIR / "outputs/.cache/figure1_bootstrap"
TABLE4_PARQUET = BASE_DIR / "outputs/table4_task_level.parquet"

# Setup joblib disk cache for bootstrap computations
memory = Memory(CACHE_DIR, verbose=0)

RELIABILITY_TASKS = [
    "HaluEval Dialog, Accuracy",
    "HaluEval QA, Accuracy",
    "HaluEval Summarization, Accuracy",
    "MemoTrap, Accuracy",
    "TruthfulQA MC1, Accuracy",
    "TruthfulQA MC2, Accuracy",
]

MODEL_WHITELIST = [
    "Repro LoRA R32",  # P=1 baseline that gets mapped to ND-LoRA [OptC9]
    "ND-LoRA [OptC9]",  # P=2, 4, 8 models
]

logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Task name mapping from lm-eval format to display format
TASK_NAME_MAP = {
    'halueval_dialogue': 'HaluEval Dialog, Accuracy',
    'halueval_qa': 'HaluEval QA, Accuracy',
    'halueval_summarization': 'HaluEval Summarization, Accuracy',
    'memo-trap_v2': 'MemoTrap, Accuracy',
    'truthfulqa_mc1': 'TruthfulQA MC1, Accuracy',
    'truthfulqa_mc2': 'TruthfulQA MC2, Accuracy',
}

# Mapping from full treatment names to abbreviated names for joining
TREATMENT_NAME_MAP = {
    "ParControl Q0.5B P=1: Repro LoRA R32": "ND-LoRA [OptC9]",
    "ParControl Q0.5B P=2: ND-LoRA [OptC9]": "ND-LoRA [OptC9]",
    "ParControl Q0.5B P=4: ND-LoRA [OptC9]": "ND-LoRA [OptC9]",
    "ParControl Q0.5B P=8: ND-LoRA [OptC9]": "ND-LoRA [OptC9]",
}


@memory.cache
def load_sample_level_data(base_path: Path) -> pd.DataFrame:
    """
    Load per-sample evaluation scores from JSON files.

    Args:
        base_path: Path to ParControl eval directory (contains P=1, P=2, P=4, P=8 subdirs)
        reliability_tasks: List of task names in display format
        model_list: List of model treatment names to filter

    Returns:
        DataFrame with columns: [base, treatment, task, P, doc_id, acc]
    """
    logger.info("Loading sample-level data from %s (cache miss - reading from disk)", base_path)
    assert base_path.exists(), f"Base path does not exist: {base_path}"

    # Invert task name mapping for lookup
    task_map_inv = {v: k for k, v in TASK_NAME_MAP.items()}
    eval_task_names = [task_map_inv[t] for t in RELIABILITY_TASKS if t in task_map_inv]

    sample_data = []
    p_values = [1, 2, 4, 8]

    for p_val in p_values:
        p_dir = base_path / f"P={p_val}"
        json_files = p_dir.glob("**/results_*.json")
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            # Extract model name from config
            model_raw = data['config']['model_name']
            model_name = MODEL_NAMES[model_raw].split(":")[-1].strip()
            if model_name not in MODEL_WHITELIST:
                continue

            for eval_task_name in eval_task_names:
                if eval_task_name not in data.get('samples', {}):
                    continue

                samples = data['samples'][eval_task_name]
                assert len(samples) > 0, f"Empty samples for {eval_task_name} in {json_file.name}"

                display_task_name = TASK_NAME_MAP[eval_task_name]

                # Extract per-sample scores using vectorized operations
                sample_df = pd.DataFrame([{'doc_id': s['doc_id'], 'acc': s['acc']} for s in samples])
                sample_df['base'] = 'Q0.5B'

                sample_df['task'] = display_task_name
                sample_df['P'] = p_val
                sample_df['run_id'] = json_file.parent.name

                if model_name == "Repro LoRA R32":
                    # Create copies for each model in whitelist so P=1 data is available for all
                    for target_model in MODEL_WHITELIST:
                        copied_df = sample_df.copy()
                        copied_df['treatment'] = target_model
                        sample_data.append(copied_df)
                        logger.info("Loaded %d samples for P=%d, task=%s, treatment=%s (from Repro LoRA R32)",
                                    len(copied_df), p_val, display_task_name, target_model)
                else:
                    sample_df['treatment'] = model_name
                    sample_data.append(sample_df)
                    logger.info("Loaded %d samples for P=%d, task=%s, treatment=%s",
                                len(sample_df), p_val, display_task_name, model_name)

    assert len(sample_data) > 0, "No sample data loaded"
    samples_df = pd.concat(sample_data, ignore_index=True)

    # Validate data structure
    required_cols = ['base', 'treatment', 'task', 'P', 'doc_id', 'acc']
    assert all(col in samples_df.columns for col in required_cols), f"Missing required columns: {required_cols}"
    assert samples_df['acc'].notna().all(), "Found NaN values in acc column"
    assert samples_df['P'].isin(p_values).all(), f"Invalid P values found: {samples_df['P'].unique()}"

    logger.info("Loaded total %d samples across %d P values, %d tasks",
                len(samples_df), samples_df['P'].nunique(), samples_df['task'].nunique())

    return samples_df


def process_task_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process task data to find optimal P and compute relative performance within each [base, treatment, task] group."""
    logger.info("Processing task data to find optimal P values within [base, treatment, task] groups")

    df_with_p = df.copy()

    # Extract P values from multi-level index
    index_df = pd.DataFrame(df_with_p.index.tolist(), index=df_with_p.index)
    p_values_extracted = index_df.iloc[:, 1].astype(str).str.extract(r'P=(\d+)')[0]
    df_with_p['P'] = pd.to_numeric(p_values_extracted, errors='coerce')

    valid_count = df_with_p['P'].notna().sum()
    total_count = len(df_with_p)
    logger.info("Successfully extracted P values from %d/%d rows", valid_count, total_count)

    df_with_p = df_with_p.dropna(subset=['P'])
    df_with_p['P'] = df_with_p['P'].astype(int)
    assert len(df_with_p) > 0, "No valid P values found after extraction"

    p_values = sorted(df_with_p['P'].unique())
    logger.info("Found P values: %s", p_values)

    # Add base and treatment from index
    df_with_p['base'] = [idx[0] for idx in df_with_p.index]
    df_with_p['treatment'] = [idx[2] for idx in df_with_p.index]

    # Add pseudo treatments: duplicate Repro LoRA R32 data for each model in MODEL_WHITELIST
    # This ensures P=1 baseline is available for all model series
    concatenable = []
    repro_data = df_with_p[df_with_p['treatment'] == "Repro LoRA R32"].copy()
    if len(repro_data) > 0:
        for target_model in MODEL_WHITELIST:
            pseudo_data = repro_data.copy()
            pseudo_data["treatment"] = target_model
            concatenable.append(pseudo_data)
            logger.info("Added pseudo treatment '%s' using Repro LoRA R32 data", target_model)

    if concatenable:
        df_with_p = pd.concat([df_with_p] + concatenable, ignore_index=True)

    # Create traces for each [base, treatment, task] group
    trace_data = []
    for task_col in df.columns:
        logger.info("Processing task: %s", task_col)

        # Group by [base, treatment] and find optimal P within each group
        for (base, treatment), group_df in df_with_p.groupby(['base', 'treatment']):
            p_performance = group_df.groupby('P')[task_col].max().dropna()

            if len(p_performance) < 2:
                continue

            optimal_p = p_performance.idxmax()
            optimal_score = p_performance.max()

            # Get P=1 baseline score for normalization
            p1_score = p_performance.get(1, None)
            if p1_score is None:
                logger.warning("No P=1 score found for task %s, base %s, treatment %s - skipping",
                               task_col, base, treatment)
                continue

            for p_val, score in p_performance.items():
                trace_data.append({
                    'base': base, 'treatment': treatment, 'task': task_col,
                    'P': p_val, 'optimal_P': optimal_p, 'delta_P': p_val - optimal_p,
                    'score': score, 'optimal_score': optimal_score, 'p1_score': p1_score,
                    'relative_performance': score / p1_score
                })

            logger.info("Task %s, base %s, treatment %s: optimal P=%d, score=%.3f, P=1 baseline=%.3f",
                        task_col, base, treatment, optimal_p, optimal_score, p1_score)

    traces_df = pd.DataFrame(trace_data)
    assert len(traces_df) > 0, "No valid traces generated"
    return traces_df


def smooth_lowess(x_data: np.ndarray, y_data: np.ndarray, x_grid: np.ndarray, frac: float = 0.4) -> np.ndarray:
    """
    Apply LOWESS smoothing and interpolate to grid.

    Args:
        x_data: Data x coordinates, shape (n_data,), must be sorted
        y_data: Data y values, shape (n_data,)
        x_grid: Evaluation points, shape (n_grid,)
        frac: LOWESS smoothing fraction

    Returns:
        Smoothed y values at x_grid points, shape (n_grid,)
    """
    assert len(x_data) >= 3, f"Need ≥3 data points for LOWESS, got {len(x_data)}"
    lowess_result = lowess(y_data, x_data, frac=frac, return_sorted=False)
    return np.interp(x_grid, x_data, lowess_result)


def smooth_triangular(x_data: np.ndarray, y_data: np.ndarray, x_grid: np.ndarray, window_size: float = 3.0) -> np.ndarray:
    """
    Apply triangular kernel smoothing using vectorized operations.

    Args:
        x_data: Data x coordinates, shape (n_data,)
        y_data: Data y values, shape (n_data,)
        x_grid: Evaluation points, shape (n_grid,)
        window_size: Kernel half-width (weight goes to 0 at ±window_size)

    Returns:
        Smoothed y values at x_grid points, shape (n_grid,)
    """
    assert len(x_data) >= 2, f"Need ≥2 data points for triangular smoothing, got {len(x_data)}"

    # Vectorized computation: distances[i, j] = |x_grid[i] - x_data[j]|
    distances = np.abs(x_grid[:, np.newaxis] - x_data[np.newaxis, :])

    # Triangular kernel: weight = max(0, 1 - distance/window_size)
    weights = np.maximum(0, 1 - distances / window_size)

    # Normalize weights (each row sums to 1)
    weight_sums = weights.sum(axis=1, keepdims=True)
    assert (weight_sums > 0).all(), "Some grid points have zero weight sum - increase window_size"
    weights_normalized = weights / weight_sums

    # Weighted average: y_smooth[i] = sum_j(weights_normalized[i,j] * y_data[j])
    return weights_normalized @ y_data


def smooth_asymmetric(x_data: np.ndarray, y_data: np.ndarray, x_grid: np.ndarray, window_size: float = 1.25) -> np.ndarray:
    """
    Apply asymmetric triangular kernel smoother that biases toward zero (optimal point).

    Uses three triangular kernels with weights that transition smoothly:
    - Far from zero: 10% away, 50% current, 40% toward zero (strong bias to optimal)
    - At zero: symmetric (all kernels centered at 0, weights become equal)

    The asymmetry (both kernel offsets and weight differences) reduces linearly as we approach zero.

    Args:
        x_data: Data x coordinates, shape (n_data,)
        y_data: Data y values, shape (n_data,)
        x_grid: Evaluation points, shape (n_grid,)
        window_size: Triangular kernel half-width (default: 1.25)

    Returns:
        Smoothed y values at x_grid points, shape (n_grid,)
    """
    assert len(x_data) >= 2, f"Need ≥2 data points for asymmetric smoothing, got {len(x_data)}"

    # Vectorized computation: signed distances from each grid point to each data point
    signed_distances = x_grid[:, np.newaxis] - x_data[np.newaxis, :]
    abs_distances = np.abs(signed_distances)

    x_grid_abs = np.abs(x_grid)
    x_grid_sign = np.sign(x_grid)

    # Asymmetry factor: 1.0 when far from zero, 0.0 at zero
    # This controls both kernel offsets AND weight differences
    asymmetry_factor = np.minimum(x_grid_abs / window_size, 1.0)

    # Kernel 1: Centered at current point (base weight 50%)
    kernel_current = np.maximum(0, 1 - abs_distances / window_size)

    # Kernel 2: Centered "toward zero"
    # Offset scales with asymmetry_factor
    offset_toward = -x_grid_sign * asymmetry_factor * window_size
    distances_from_toward = np.abs(signed_distances - offset_toward[:, np.newaxis])
    kernel_toward = np.maximum(0, 1 - distances_from_toward / window_size)

    # Kernel 3: Centered "away from zero"
    # Offset scales with asymmetry_factor
    offset_away = x_grid_sign * asymmetry_factor * window_size
    distances_from_away = np.abs(signed_distances - offset_away[:, np.newaxis])
    kernel_away = np.maximum(0, 1 - distances_from_away / window_size)

    # Weight transitions: at zero, all weights equal (symmetric)
    # Far from zero: 10% away, 50% current, 40% toward
    # Weight difference from symmetric (1/3 each) scales with asymmetry_factor
    w_current = 0.5  # stays constant
    w_toward = (1/3) + asymmetry_factor[:, np.newaxis] * (0.4 - 1/3)  # 1/3 → 0.4
    w_away = (1/3) - asymmetry_factor[:, np.newaxis] * (1/3 - 0.1)    # 1/3 → 0.1

    # Combine kernels with position-dependent weights
    weights = w_current * kernel_current + w_toward * kernel_toward + w_away * kernel_away

    # Normalize weights (each row sums to 1)
    weight_sums = weights.sum(axis=1, keepdims=True)

    # Handle grid points with zero weight (fallback to nearest neighbor)
    zero_weight_mask = (weight_sums == 0).flatten()
    if zero_weight_mask.any():
        logger.warning("Found %d grid points with zero weight, using nearest neighbor", zero_weight_mask.sum())
        nearest_indices = np.argmin(abs_distances[zero_weight_mask], axis=1)
        weights[zero_weight_mask] = 0
        weights[zero_weight_mask, nearest_indices] = 1
        weight_sums = weights.sum(axis=1, keepdims=True)

    weights_normalized = weights / weight_sums

    # Weighted average: y_smooth[i] = sum_j(weights_normalized[i,j] * y_data[j])
    return weights_normalized @ y_data


def load_table4_data() -> pd.DataFrame:
    """Load table4 parquet and return DataFrame with P index and mean values across tasks."""
    df = pd.read_parquet(TABLE4_PARQUET)
    return df.groupby('P').mean()


def get_D_spec_for_P(P_values: np.ndarray) -> np.ndarray:
    """Map P values to their corresponding spectral diversity D_spec from table4 parquet."""
    table4_df = load_table4_data()
    D_spec_by_P = table4_df['dspec_cosine']
    # P=1 uses P=2 value (single stream has no diversity to measure)
    D_spec_map = {1: D_spec_by_P.get(2, D_spec_by_P.iloc[0])} | D_spec_by_P.to_dict()
    # Sort for np.interp
    sorted_P = np.array(sorted(D_spec_map.keys()))
    sorted_D = np.array([D_spec_map[p] for p in sorted_P])
    D_spec_array = np.array([D_spec_map.get(int(p), np.interp(p, sorted_P, sorted_D)) for p in P_values])
    logger.info("Mapped P values %s to D_spec values %s", P_values, D_spec_array)
    return D_spec_array


def compute_theoretical_performance_thm1(P_values: np.ndarray, D_spec, C_star: float,
                                         snr: float) -> np.ndarray:
    """
    Compute (1-P(H)) / (1-P(H|P=1)) using Theorem 1's hallucination bound.

    Define v(P) = (1 - C_* * D) / P + C_* * D, so v(1) = 1.
    From Theorem 1: P(H) ≤ v(P) / (v(P) + SNR), hence 1 - P(H) ≥ SNR / (v(P) + SNR).
    The reliability ratio is: (1 - P(H)) / (1 - P(H|P=1)) = (v(1) + SNR) / (v(P) + SNR) = (1 + SNR) / (v(P) + SNR).

    Args:
        P_values: Array of P values to evaluate
        D_spec: Spectral diversity index (scalar or array)
        C_star: Correlation scaling factor
        snr: Signal-to-noise ratio

    Returns:
        Array of (1-P(H)) / (1-P(H|P=1)) values (comparable to eval_score / eval_score_P1)
    """
    D = np.asarray(D_spec) if isinstance(D_spec, (list, np.ndarray)) else np.full_like(P_values, D_spec, dtype=float)
    v_P = (1 - C_star * D) / P_values + C_star * D
    return (1 + snr) / (v_P + snr)


def compute_theoretical_performance_thm2(P_values: np.ndarray, rho_0: float, beta: float,
                                         gamma: float, signal_to_noise: float) -> np.ndarray:
    """
    Compute theoretical relative performance using Theorem 2's U-shaped hallucination bound.

    From Theorem 2: P(H) ≤ B(P) = σ²g(P) / (σ²g(P) + μ²)
    where g(P) = (1-ρ(P))/P + ρ(P) and ρ(P) = ρ₀ + β(P-1)^γ

    Since accuracy ∝ 1 - P(H), relative performance = (g(1) + SNR) / (g(P) + SNR)
    where SNR = μ²/σ² is the signal-to-noise ratio.

    Args:
        P_values: Array of P values to evaluate
        rho_0: Baseline correlation at P=1
        beta: Correlation growth coefficient (β > 0)
        gamma: Correlation growth exponent (γ > 0)
        signal_to_noise: Signal-to-noise ratio μ²/σ²

    Returns:
        Array of relative performance values normalized by P=1 baseline
    """
    assert rho_0 >= 0 and rho_0 <= 1, f"rho_0 must be in [0,1], got {rho_0}"
    assert beta >= 0, f"beta must be non-negative, got {beta}"
    assert gamma > 0, f"gamma must be positive, got {gamma}"
    assert signal_to_noise > 0, f"signal_to_noise must be positive, got {signal_to_noise}"
    assert (P_values >= 1).all(), f"P values must be >= 1, got {P_values}"

    # Correlation function: ρ(P) = ρ₀ + β(P-1)^γ
    rho_P = rho_0 + beta * (P_values - 1) ** gamma

    # Check correlation stays in valid range
    assert (rho_P <= 1.0).all(), f"Correlation exceeds 1.0 for some P values: max={rho_P.max()}"

    # Variance scaling factor: g(P) = (1-ρ(P))/P + ρ(P)
    g_P = (1 - rho_P) / P_values + rho_P

    # Baseline at P=1: g(1) = (1-ρ₀)/1 + ρ₀ = 1
    g_1 = 1.0

    # Relative performance from Theorem 2 bound: (g(1) + SNR) / (g(P) + SNR)
    relative_performance = (g_1 + signal_to_noise) / (g_P + signal_to_noise)

    return relative_performance


def fit_theoretical_params_thm1(empirical_means: pd.Series, peak_weight: float = 1.0) -> tuple:
    """
    Fit C_* and SNR to match empirical P(H)/P(H|P=1) curve using Theorem 1.

    Args:
        empirical_means: Series with index=P values and values=P(H)/P(H|P=1)
        peak_weight: Weight for peak constraint

    Returns:
        Tuple of (C_star, snr, r_squared)
    """
    assert len(empirical_means) >= 2, f"Need at least 2 P values to fit, got {len(empirical_means)}"
    P_values = empirical_means.index.values.astype(float)
    empirical_perf = empirical_means.values
    D_spec_values = get_D_spec_for_P(P_values)

    peak_idx = np.argmax(empirical_perf)  # maximum (1-P(H)) is best
    peak_P, peak_value = P_values[peak_idx], empirical_perf[peak_idx]
    peak_D_spec = get_D_spec_for_P(np.array([peak_P]))

    def mse_loss(params):
        C_star, snr = params
        if C_star < 0 or C_star > 2 or snr <= 0:
            return 1e10
        theoretical = compute_theoretical_performance_thm1(P_values, D_spec_values, C_star, snr)
        mse = np.mean((theoretical - empirical_perf) ** 2)
        theoretical_peak = compute_theoretical_performance_thm1(np.array([peak_P]), peak_D_spec, C_star, snr)[0]
        return mse + peak_weight * (theoretical_peak - peak_value) ** 2

    result = minimize(mse_loss, x0=[1.0, 10.0], method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    assert result.success, f"Optimization failed: {result.message}"

    C_star_fitted, snr_fitted = result.x
    theoretical_fitted = compute_theoretical_performance_thm1(P_values, D_spec_values, C_star_fitted, snr_fitted)
    ss_res = np.sum((empirical_perf - theoretical_fitted) ** 2)
    ss_tot = np.sum((empirical_perf - np.mean(empirical_perf)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    logger.info("Fitted C_*=%.4f, SNR=%.4f with R²=%.4f", C_star_fitted, snr_fitted, r_squared)
    return C_star_fitted, snr_fitted, r_squared


def fit_theoretical_params_thm2(empirical_means: pd.Series, peak_weight: float = 1.0) -> tuple:
    """
    Fit β, γ, and signal-to-noise ratio to match empirical ND-LoRA mean performance curve using Theorem 2.

    Uses Theorem 2's correlation model: ρ(P) = β(P-1)^γ with ρ₀=0 fixed
    (at P=1 there's only one stream, so no inter-stream correlation exists).

    Args:
        empirical_means: Series with index=P values and values=mean relative performance
        peak_weight: Weight for peak constraint (higher = force better match at optimal P)

    Returns:
        Tuple of (rho_0, beta, gamma, signal_to_noise) that minimize MSE between theory and empirical data
    """
    assert len(empirical_means) >= 3, f"Need at least 3 P values to fit 3 parameters, got {len(empirical_means)}"
    P_values = empirical_means.index.values
    empirical_perf = empirical_means.values

    # Find peak value (optimal P)
    peak_idx = np.argmax(empirical_perf)
    peak_P = P_values[peak_idx]
    peak_value = empirical_perf[peak_idx]

    # Fix ρ₀ = 0 (no correlation at P=1 since there's only one stream)
    rho_0 = 0.0

    def mse_loss(params):
        """Mean squared error between theoretical and empirical performance with peak constraint."""
        beta, gamma, signal_to_noise = params

        # Bounds checking
        if beta < 0 or beta > 1:  # Correlation growth coefficient
            return 1e10
        if gamma <= 0 or gamma > 5:  # Reasonable range for exponent
            return 1e10
        if signal_to_noise <= 0 or signal_to_noise > 1000:
            return 1e10

        # Check correlation doesn't exceed 1.0 for any P in range
        rho_max = rho_0 + beta * (P_values.max() - 1) ** gamma
        if rho_max > 1.0:
            return 1e10

        theoretical_perf = compute_theoretical_performance_thm2(P_values, rho_0, beta, gamma, signal_to_noise)

        # Standard MSE across all points
        mse = np.mean((theoretical_perf - empirical_perf) ** 2)

        # Peak constraint: heavily penalize mismatch at optimal P
        theoretical_peak = compute_theoretical_performance_thm2(np.array([peak_P]), rho_0, beta, gamma, signal_to_noise)[0]
        peak_penalty = peak_weight * (theoretical_peak - peak_value) ** 2

        return mse + peak_penalty

    # Initial guess: β=0.15, γ=0.63, SNR=10.0
    # With γ=0.63, β=0.15 gives ρ(4) = 0.15 * 3^0.63 ≈ 0.30 (reasonable correlation)
    result = minimize(mse_loss, x0=[0.15, 0.63, 10.0], method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
    assert result.success, f"Optimization failed: {result.message}"

    beta_fitted, gamma_fitted, snr_fitted = result.x
    final_mse = result.fun

    # Verify peak matching and compute R²
    theoretical_peak_fitted = compute_theoretical_performance_thm2(np.array([peak_P]), rho_0,
                                                                   beta_fitted, gamma_fitted, snr_fitted)[0]
    theoretical_perf_fitted = compute_theoretical_performance_thm2(P_values, rho_0, beta_fitted, gamma_fitted, snr_fitted)

    # Calculate R²: R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((empirical_perf - theoretical_perf_fitted) ** 2)
    ss_tot = np.sum((empirical_perf - np.mean(empirical_perf)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute resulting rho(4) for reference
    rho_4_result = rho_0 + beta_fitted * 3**gamma_fitted

    logger.info("Fitted β=%.4f, γ=%.4f, SNR=%.4f with MSE=%.6f, R²=%.4f (ρ₀=0 fixed, peak_weight=%.1f)",
                beta_fitted, gamma_fitted, snr_fitted, final_mse, r_squared, peak_weight)
    logger.info("Peak match: empirical=%.6f, theoretical=%.6f, error=%.6f",
                peak_value, theoretical_peak_fitted, theoretical_peak_fitted - peak_value)
    logger.info("Resulting correlation at P=4: ρ(4)=%.4f (note: D_spec=0.3755 is a different quantity)",
                rho_4_result)

    return rho_0, beta_fitted, gamma_fitted, snr_fitted, r_squared


@memory.cache
def bootstrap_smooth(traces_df: pd.DataFrame, samples_df: pd.DataFrame, smoother_method: str,
                     smoother_param: float = None, n_boot: int = 1000, xlim=(-3, 3), seed: int = 42) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals using arbitrary smoothing function on resampled data.

    Strategy: For each bootstrap iteration:
    1. Resample samples within each task
    2. For each task: compute mean relative_acc by delta_P, then smooth
    3. Store smoothed traces in wide format

    Args:
        traces_df: DataFrame with aggregated trace data including delta_P and optimal_score
        samples_df: DataFrame with per-sample scores [base, treatment, task, P, doc_id, acc]
        smoother_method: Smoothing method name ('lowess', 'triangular', or 'asymmetric')
        smoother_param: Parameter for smoothing function (None=use function default)
        n_boot: Number of bootstrap iterations
        xlim: Tuple of (xmin, xmax) for x-axis limits
        seed: Random seed for reproducibility

    Returns:
        DataFrame with index=delta_P (x-values) and MultiIndex columns=[task, bootstrap_ix]
    """
    # Smoothing function registry
    smoothing_registry = {
        'lowess': smooth_lowess,
        'triangular': smooth_triangular,
        'asymmetric': smooth_asymmetric,
    }
    assert smoother_method in smoothing_registry, \
        f"Unknown smoothing method: {smoother_method}. Choose from {list(smoothing_registry.keys())}"
    smoother_func = smoothing_registry[smoother_method]

    logger.info("Computing task-level bootstrap with %s (n_boot=%d, param=%s, cache miss)",
                smoother_method, n_boot, smoother_param)

    # Merge to add delta_P and optimal info to samples
    merged_df = traces_df.merge(samples_df, on=['base', 'treatment', 'task', 'P'], how='inner')
    assert len(merged_df) > 0, "No samples found after merge"
    logger.info("Merged %d samples with delta_P information", len(merged_df))

    # Normalize each sample's accuracy by its task's P=1 baseline score
    merged_df['relative_acc'] = merged_df['acc'] / merged_df['p1_score']
    logger.info("Normalized sample accuracies by P=1 baseline score within each (base, treatment, task) group")

    tasks = merged_df['task'].unique()
    logger.info("Found %d tasks: %s", len(tasks), tasks)

    x_grid = np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) * 100 + 1))
    bootstrap_df = pd.DataFrame(columns=pd.MultiIndex.from_product([tasks, np.arange(n_boot)], names=['task', 'boot_ix']),
                                index=x_grid, dtype=float)
    bootstrap_df.index.name = 'delta_P'

    np.random.seed(seed)
    nargs = [smoother_param] if smoother_param is not None else []
    for i in range(n_boot):
        # NOTE: Process each task separately, as not all delta_Ps are sampled for each task and
        # it'll create weird dicontinuities otherwise.
        for task in tasks:
            task_df = merged_df[merged_df['task'] == task]
            boot_sample = task_df.sample(n=len(task_df), replace=True, random_state=seed+i)
            boot_means = boot_sample.groupby('delta_P', sort=True)['relative_acc'].mean()
            bootstrap_df[(task, i)] = smoother_func(boot_means.index.values, boot_means.values, x_grid, *nargs)

        if (i + 1) % 100 == 0:
            logger.info("Completed %d/%d bootstrap iterations", i + 1, n_boot)

    logger.info("Created DataFrame with shape %s (index=%d x_values, columns=%d tasks × %d bootstrap_iters)",
                bootstrap_df.shape, len(x_grid), len(tasks), n_boot)

    return bootstrap_df


def generate_optimality_plot(traces_df: pd.DataFrame, samples_df: pd.DataFrame, output_dir: Path,
                             smoothing_method: str = 'lowess', smoothing_param: float = None, n_boot: int = 1000,
                             discretize: bool = True, xlim: tuple = (-3, 3), theoretical_fit: str = 'thm2'):
    """
    Generate optimality plot with empirical bootstrap curves and optional theoretical predictions.

    Args:
        traces_df: DataFrame with trace data
        samples_df: DataFrame with sample-level data
        output_dir: Directory to save output figures
        smoothing_method: One of ['lowess', 'triangular', 'asymmetric']
        smoothing_param: Smoothing parameter (frac for lowess, window_size for triangular/asymmetric)
        n_boot: Number of bootstrap iterations
        discretize: If True, filter to integer delta_P values; if False, plot full smooth curve
        xlim: Tuple of (xmin, xmax) for x-axis limits
        theoretical_fit: One of ['thm1', 'thm2', 'none'] for theoretical overlay
    """
    # Smoothing method registry: maps method name to display label
    smoothing_display_labels = {
        'lowess': 'Bootstrapped LOWESS',
        'triangular': 'Bootstrapped Kernel',
        'asymmetric': 'Bootstrapped Kernel',
    }
    assert smoothing_method in smoothing_display_labels, \
        f"Unknown smoothing method: {smoothing_method}. Choose from {list(smoothing_display_labels.keys())}"

    method_label = smoothing_display_labels[smoothing_method]
    bootstrap_df = bootstrap_smooth(traces_df, samples_df, smoother_method=smoothing_method,
                                    smoother_param=smoothing_param, n_boot=n_boot, xlim=xlim,
                                    seed=42)

    # Average across tasks: for each bootstrap_ix, average all tasks
    # Result: DataFrame with index=delta_P, columns=bootstrap_ix
    averaged_df = bootstrap_df.groupby(level='boot_ix', axis=1).mean()

    # Compute pointwise statistics across bootstrap samples
    stats_df = pd.DataFrame({
        'delta_P': averaged_df.index,
        'mean': averaged_df.mean(axis=1),
        'p10': averaged_df.quantile(0.10, axis=1),
        'p90': averaged_df.quantile(0.90, axis=1)
    })

    if discretize:
        # Filter to only exact integer delta_P values
        is_integer = np.abs(stats_df['delta_P'] - np.round(stats_df['delta_P'])) < 1e-9
        plot_df = stats_df[is_integer].copy()
        plot_df['delta_P'] = plot_df['delta_P'].astype(int)
        logger.info("Discretized to %d integer delta_P values: %s", len(plot_df), plot_df['delta_P'].values)
    else:
        plot_df = stats_df
        logger.info("Using full smooth curve with %d points", len(plot_df))

    logger.info("Mean curve range: [%.3f, %.3f]", plot_df['mean'].min(), plot_df['mean'].max())
    zero_idx = (plot_df['delta_P'] - 0).abs().idxmin()
    zero_row = plot_df.loc[zero_idx]
    logger.info("CI width at delta_P=0: %.3f", zero_row['p90'] - zero_row['p10'])

    # Compute theoretical prediction (if enabled)
    theoretical_df = None
    theory_label = None
    if theoretical_fit != 'none':
        # Find empirical optimal P from raw data to establish coordinate system
        nd_lora_traces = traces_df[traces_df['treatment'] == 'ND-LoRA [OptC9]']
        empirical_by_P_raw = nd_lora_traces.groupby('P')['relative_performance'].mean()
        optimal_P_empirical = empirical_by_P_raw.idxmax()
        logger.info("Empirical optimal P for ND-LoRA: P=%d", optimal_P_empirical)

        # Fit to smoothed bootstrap data at absolute integer P values within visible range
        # Convert deltaP back to absolute P for each point in the smooth curve
        plot_df_with_P = plot_df.copy()
        plot_df_with_P['P'] = plot_df_with_P['delta_P'] + optimal_P_empirical

        # Extract values at integer P positions within visible range (xlim bounds)
        P_min_visible = int(np.ceil(optimal_P_empirical + xlim[0]))
        P_max_visible = int(np.floor(optimal_P_empirical + xlim[1]))
        P_integers = np.arange(P_min_visible, P_max_visible + 1)
        empirical_for_fit = pd.Series(
            index=P_integers,
            data=np.interp(P_integers, plot_df_with_P['P'].values, plot_df_with_P['mean'].values)
        )
        logger.info("Fitting on %d interpolated absolute P values within visible range [%d, %d]: %s",
                    len(empirical_for_fit), P_min_visible, P_max_visible, list(P_integers))
        logger.info("Interpolated performance values: %s",
                    {p: f"{v:.4f}" for p, v in empirical_for_fit.items()})

        if theoretical_fit == 'thm1':
            logger.info("Fitting Theorem 1 using %d interpolated P values", len(empirical_for_fit))
            C_star_fitted, snr_fitted, r_squared = fit_theoretical_params_thm1(empirical_for_fit)
            theory_label = f'Theoretical Prediction ($R^2$={r_squared:.3f})'

        elif theoretical_fit == 'thm2':
            assert len(empirical_for_fit) >= 3, f"Need at least 3 P values for Theorem 2, got {len(empirical_for_fit)}"
            logger.info("Fitting Theorem 2 using %d interpolated P values", len(empirical_for_fit))
            rho_0_fitted, beta_fitted, gamma_fitted, snr_fitted, r_squared = fit_theoretical_params_thm2(empirical_for_fit)
            theory_label = f'Theoretical Prediction ($R^2$={r_squared:.3f})'

        # Evaluate theoretical model on FULL plot range
        plot_df_full = plot_df.copy()
        plot_df_full['P'] = plot_df_full['delta_P'] + optimal_P_empirical
        P_values_eval = plot_df_full['P'].values

        if theoretical_fit == 'thm1':
            D_spec_eval = get_D_spec_for_P(P_values_eval)
            theoretical_perf = compute_theoretical_performance_thm1(P_values_eval, D_spec_eval,
                                                                    C_star_fitted, snr_fitted)
        elif theoretical_fit == 'thm2':
            theoretical_perf = compute_theoretical_performance_thm2(P_values_eval, rho_0_fitted,
                                                                    beta_fitted, gamma_fitted, snr_fitted)

        # Results are in delta_P coordinates
        theoretical_df = pd.DataFrame({
            'delta_P': plot_df_full['delta_P'].values,
            'theoretical_perf': theoretical_perf
        })
        logger.info("Theoretical predictions: %d points from delta_P in [%.1f, %.1f]",
                    len(theoretical_df), theoretical_df['delta_P'].min(), theoretical_df['delta_P'].max())

    fig, ax = plt.subplots(figsize=(5, 3))

    # Prepare scatter data: sample bootstrap curves for visualization
    # Use a subset of bootstrap samples to avoid overcrowding
    n_scatter_samples = min(50, n_boot)
    scatter_indices = np.linspace(0, n_boot - 1, n_scatter_samples, dtype=int)

    # Create scatter plot data by downsampling x-axis for visual clarity
    if discretize:
        # For discrete plots, only show scatter at integer delta_P
        scatter_x = plot_df['delta_P'].values
        scatter_data = averaged_df.loc[plot_df['delta_P'].values, scatter_indices]
    else:
        # For continuous plots, downsample to ~20 points for visual clarity
        n_scatter_x = min(20, len(averaged_df))
        scatter_x_indices = np.linspace(0, len(averaged_df) - 1, n_scatter_x, dtype=int)
        scatter_x = averaged_df.index[scatter_x_indices].values
        scatter_data = averaged_df.iloc[scatter_x_indices, scatter_indices]

    # Plot bootstrap samples as small transparent points (subtract 1 for relative improvement)
    for col_idx, boot_idx in enumerate(scatter_indices):
        ax.scatter(scatter_x, scatter_data.iloc[:, col_idx] - 1,
                   s=8, alpha=0.15, color='C0', edgecolors='none', rasterized=True)

    # Plot CI and mean (subtract 1 for relative improvement)
    kwargs = dict(marker='o', markersize=7, markeredgewidth=0.5,
                  markeredgecolor='white') if discretize else {}
    ax.plot(plot_df['delta_P'], plot_df['mean'] - 1, label=method_label,
            linewidth=2.5, color='C0', zorder=5, **kwargs)
    ax.fill_between(plot_df['delta_P'], plot_df['p10'] - 1, plot_df['p90'] - 1,
                    alpha=0.25, label='80% CI', color='C0', linewidth=0)

    # Plot theoretical prediction (if enabled, subtract 1 for relative improvement)
    if theoretical_df is not None:
        ax.plot(theoretical_df['delta_P'], theoretical_df['theoretical_perf'] - 1,
                label=theory_label, linewidth=2.0, color='darkorange', linestyle='--', marker='s',
                markersize=6, markeredgewidth=0.5, markeredgecolor='white', zorder=6)

    # Mark the optimal point at delta_P=0 and baseline at P=1
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal P', zorder=3)
    ax.axhline(y=plot_df["mean"].max() - 1, color='red', linestyle='--', alpha=0.5, linewidth=1.5, zorder=3)
    # ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='P=1', zorder=3)

    ax.set_xlabel('$\Delta P$ (Distance from Optimal Neural Diversity)', fontsize=10)
    ax.set_ylabel('Reliability', fontsize=10)
    ax.set_title("Multi-Task Reliability: Sensitivity to Neural Diversity", fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.000, 0.150)  # Relative improvement over P=1 baseline

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+g}' if x != 0 else '0'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.1%}' if y != 0 else '0'))
    ax.set_xlim(xlim)

    plt.tight_layout()

    plt.savefig(output_dir / 'figure1_optimal_diversity.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_optimal_diversity.pdf', dpi=600, bbox_inches='tight')

    logger.info("Saved figure to %s", output_dir)
    # plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate optimal diversity figure with different smoothing methods')
    parser.add_argument('--smoothing', type=str, default='lowess', choices=['lowess', 'triangular', 'asymmetric'],
                        help='Smoothing method to use (default: lowess)')
    parser.add_argument('--smoothing-param', type=float, default=None,
                        help='Smoothing parameter: frac for lowess (default: 0.4), '
                             'window_size for triangular/asymmetric (default: 3.0)')
    parser.add_argument('--n-boot', type=int, default=1000, help='Number of bootstrap samples (default: 1000)')
    parser.add_argument('--xlim', type=float, nargs=2, default=[-3, 3], metavar=('XMIN', 'XMAX'),
                        help='X-axis limits as two floats (default: -3 3)')
    parser.add_argument('--no-discretize', action='store_true',
                        help='Plot full smooth curve instead of filtering to integer delta_P values')
    parser.add_argument('--theoretical-fit', type=str, default='thm1', choices=['thm1', 'thm2', 'none'],
                        help='Theoretical model to overlay: thm1 (Theorem 1), thm2 (Theorem 2), none (default: thm2)')
    args = parser.parse_args()

    logger.info("Starting figure generation with smoothing=%s, n_boot=%d, xlim=%s, discretize=%s, theoretical_fit=%s",
                args.smoothing, args.n_boot, args.xlim, not args.no_discretize, args.theoretical_fit)

    # Load aggregated data for trace computation (regenerated on demand from raw S3 evals).
    from build_scores import ensure_scores_parquet
    df = pd.read_parquet(ensure_scores_parquet(plot_type="pub", mode="full"))
    traces_df = process_task_data(df)

    # Filter to whitelisted models and reliability tasks
    traces_df = traces_df[traces_df["treatment"].isin(MODEL_WHITELIST) &
                          traces_df["task"].isin(RELIABILITY_TASKS)]
    logger.info("Filtered traces_df to %d rows", len(traces_df))

    # Sync raw sample-level evals to EVALS_DIR on demand, then load per-sample data.
    from statsig_utils import sync_s3_to_local
    sync_s3_to_local(force=False)
    samples_df = load_sample_level_data(EVALS_DIR)
    logger.info("Loaded samples_df with %d rows", len(samples_df))

    # Generate plot with specified smoothing method
    generate_optimality_plot(traces_df, samples_df, OUTPUT_DIR, smoothing_method=args.smoothing,
                             smoothing_param=args.smoothing_param, n_boot=args.n_boot,
                             discretize=not args.no_discretize, xlim=tuple(args.xlim),
                             theoretical_fit=args.theoretical_fit)


if __name__ == '__main__':
    main()
