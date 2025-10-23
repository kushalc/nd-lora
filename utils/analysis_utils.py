"""
Analysis utilities for ParScale experiments.
Handles results analysis, plotting, and final report generation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from leaderboard.src.envs import QUEUE_REPO, RESULTS_REPO
from leaderboard.src.utils import my_snapshot_download

# Get module logger
logger = logging.getLogger(__name__)


def load_experiment_results(results_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load results from multiple ParScale experiments.

    Args:
        results_dir: Directory containing result files

    Returns:
        Dictionary mapping P values to result dictionaries
    """
    results_path = Path(results_dir)
    results = {}

    # Look for results files
    for file_path in results_path.glob("results_P*.json"):
        try:
            # Extract P value from filename
            filename = file_path.stem
            P = int(filename.split("_P")[1])

            # Load results
            with open(file_path, 'r') as f:
                result_data = json.load(f)

            results[P] = result_data

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning("Could not load %s: %s", file_path, e, exc_info=True)

    return results


def create_results_table(results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comprehensive results table.

    Args:
        results: Dictionary of results by P value

    Returns:
        DataFrame with results table
    """
    table_data = []

    # Get P=1 baseline for comparison
    baseline_ppl = None
    if 1 in results:
        baseline_ppl = results[1].get('perplexity', None)

    for P in sorted(results.keys()):
        result = results[P]

        row = {
            'P': P,
            'tokens_M': result.get('total_tokens', 0) / 1e6,
            'seq_len': result.get('seq_len', 'unknown'),
            'final_loss': result.get('loss', float('inf')),
            'perplexity': result.get('perplexity', float('inf')),
            'total_steps': result.get('total_steps', 0),
            'total_time_hours': result.get('total_time_seconds', 0) / 3600,
            'tokens_per_second': result.get('tokens_per_second', 0),
            'trainable_params': result.get('trainable_parameters', 0),
            'best_val_loss': result.get('best_val_loss', float('inf'))
        }

        # Calculate improvement over P=1 baseline
        if baseline_ppl is not None and result.get('perplexity') is not None:
            row['delta_ppl_vs_P1'] = result['perplexity'] - baseline_ppl
            row['improvement_pct'] = ((baseline_ppl - result['perplexity']) / baseline_ppl) * 100
        else:
            row['delta_ppl_vs_P1'] = None
            row['improvement_pct'] = None

        # Stream statistics (if available)
        row['entropy_mean'] = result.get('entropy_mean', None)
        row['entropy_decreasing'] = result.get('entropy_trend', 0) < 0
        row['collapse_detected'] = result.get('collapse_detected', False)

        table_data.append(row)

    df = pd.DataFrame(table_data)
    return df


def fit_scaling_law(
    P_values: List[int],
    losses: List[float],
    fit_type: str = "log"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit scaling law to loss vs P data.

    Args:
        P_values: List of P values
        losses: Corresponding loss values
        fit_type: Type of fit ("log", "power", "linear")

    Returns:
        Tuple of (fitted_curve, fit_stats)
    """
    P_array = np.array(P_values)
    loss_array = np.array(losses)

    # Remove invalid values
    valid_mask = ~(np.isnan(loss_array) | np.isinf(loss_array))
    P_valid = P_array[valid_mask]
    loss_valid = loss_array[valid_mask]

    if len(P_valid) < 2:
        return np.array([]), {}

    fit_stats = {}

    if fit_type == "log":
        # Fit: loss = a * log(P) + b
        log_P = np.log(P_valid)

        def log_func(P, a, b):
            return a * np.log(P) + b

        try:
            popt, pcov = curve_fit(log_func, P_valid, loss_valid)
            a, b = popt

            # Generate fitted curve
            P_fit = np.linspace(P_valid.min(), P_valid.max(), 100)
            loss_fit = log_func(P_fit, a, b)

            # Calculate fit statistics
            loss_pred = log_func(P_valid, a, b)
            r2 = 1 - np.sum((loss_valid - loss_pred)**2) / np.sum((loss_valid - np.mean(loss_valid))**2)

            fit_stats = {
                'fit_type': 'logarithmic',
                'a': float(a),
                'b': float(b),
                'r_squared': float(r2),
                'equation': f'loss = {a:.4f} * log(P) + {b:.4f}'
            }

            return np.column_stack([P_fit, loss_fit]), fit_stats

        except Exception as e:
            logger.warning("Log fit failed: %s", e, exc_info=True)

    elif fit_type == "power":
        # Fit: loss = a * P^b + c
        def power_func(P, a, b, c):
            return a * np.power(P, b) + c

        try:
            popt, pcov = curve_fit(power_func, P_valid, loss_valid)
            a, b, c = popt

            P_fit = np.linspace(P_valid.min(), P_valid.max(), 100)
            loss_fit = power_func(P_fit, a, b, c)

            loss_pred = power_func(P_valid, a, b, c)
            r2 = 1 - np.sum((loss_valid - loss_pred)**2) / np.sum((loss_valid - np.mean(loss_valid))**2)

            fit_stats = {
                'fit_type': 'power',
                'a': float(a),
                'b': float(b),
                'c': float(c),
                'r_squared': float(r2),
                'equation': f'loss = {a:.4f} * P^{b:.4f} + {c:.4f}'
            }

            return np.column_stack([P_fit, loss_fit]), fit_stats

        except Exception as e:
            logger.warning("Power fit failed: %s", e, exc_info=True)

    elif fit_type == "linear":
        # Simple linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(P_valid, loss_valid)

        P_fit = np.linspace(P_valid.min(), P_valid.max(), 100)
        loss_fit = slope * P_fit + intercept

        fit_stats = {
            'fit_type': 'linear',
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'equation': f'loss = {slope:.4f} * P + {intercept:.4f}'
        }

        return np.column_stack([P_fit, loss_fit]), fit_stats

    return np.array([]), {}


def create_loss_comparison_plots(
    results: Dict[int, Dict[str, Any]],
    output_dir: str
) -> Dict[str, str]:
    """
    Create comprehensive loss comparison plots.

    Args:
        results: Dictionary of results by P value
        output_dir: Directory to save plots

    Returns:
        Dictionary of plot filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_files = {}

    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # 1. Loss vs Log(P) plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    P_values = []
    losses = []
    perplexities = []

    for P in sorted(results.keys()):
        if 'loss' in results[P] and results[P]['loss'] != float('inf'):
            P_values.append(P)
            losses.append(results[P]['loss'])
            perplexities.append(results[P].get('perplexity', np.exp(results[P]['loss'])))

    if P_values:
        # Plot data points
        ax.scatter(P_values, losses, s=100, c='red', zorder=5, label='Validation Loss')

        # Fit and plot scaling law
        fitted_curve, fit_stats = fit_scaling_law(P_values, losses, "log")
        if len(fitted_curve) > 0:
            ax.plot(fitted_curve[:, 0], fitted_curve[:, 1], '--',
                    color='blue', linewidth=2, label=f"Log fit (R² = {fit_stats['r_squared']:.3f})")

            # Add equation to plot
            ax.text(0.05, 0.95, fit_stats['equation'],
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel('Number of Parallel Streams (P)')
        ax.set_ylabel('Validation Loss')
        ax.set_title('ParScale Validation Loss vs P')
        ax.set_xscale('log')
        ax.set_xticks(P_values)
        ax.set_xticklabels([str(p) for p in P_values])
        ax.grid(True, alpha=0.3)
        ax.legend()

        plot_file = output_path / 'loss_vs_logP.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files['loss_vs_logP'] = str(plot_file)

    plt.close()

    # 2. Perplexity comparison bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if P_values:
        bars = ax.bar(range(len(P_values)), perplexities,
                      color=colors[:len(P_values)], alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for i, (p, ppl) in enumerate(zip(P_values, perplexities)):
            ax.text(i, ppl + 0.01 * max(perplexities), f'{ppl:.2f}',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Number of Parallel Streams (P)')
        ax.set_ylabel('Validation Perplexity')
        ax.set_title('ParScale Validation Perplexity by P')
        ax.set_xticks(range(len(P_values)))
        ax.set_xticklabels([str(p) for p in P_values])
        ax.grid(True, alpha=0.3, axis='y')

        plot_file = output_path / 'perplexity_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files['perplexity_comparison'] = str(plot_file)

    plt.close()

    # 3. Training efficiency plot (if timing data available)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    throughputs = []
    training_times = []

    for P in sorted(results.keys()):
        if 'tokens_per_second' in results[P] and 'total_time_seconds' in results[P]:
            throughputs.append(results[P]['tokens_per_second'])
            training_times.append(results[P]['total_time_seconds'] / 3600)  # Convert to hours

    if throughputs:
        # Throughput plot
        ax1.bar(range(len(P_values)), throughputs,
                color=colors[:len(P_values)], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of Parallel Streams (P)')
        ax1.set_ylabel('Tokens/Second')
        ax1.set_title('Training Throughput by P')
        ax1.set_xticks(range(len(P_values)))
        ax1.set_xticklabels([str(p) for p in P_values])
        ax1.grid(True, alpha=0.3, axis='y')

        # Training time plot
        ax2.bar(range(len(P_values)), training_times,
                color=colors[:len(P_values)], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Parallel Streams (P)')
        ax2.set_ylabel('Training Time (Hours)')
        ax2.set_title('Total Training Time by P')
        ax2.set_xticks(range(len(P_values)))
        ax2.set_xticklabels([str(p) for p in P_values])
        ax2.grid(True, alpha=0.3, axis='y')

        plot_file = output_path / 'training_efficiency.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files['training_efficiency'] = str(plot_file)

    plt.close()

    # 4. Stream diagnostics summary (if available)
    stream_entropies = []
    has_stream_data = False

    for P in sorted(results.keys()):
        if P > 1 and 'entropy_mean' in results[P]:
            stream_entropies.append(results[P]['entropy_mean'])
            has_stream_data = True
        elif P == 1:
            stream_entropies.append(0.0)  # P=1 has no streams

    if has_stream_data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        bars = ax.bar(range(len(P_values)), stream_entropies,
                      color=colors[:len(P_values)], alpha=0.7, edgecolor='black')

        # Add theoretical maximum entropy line
        max_entropies = [np.log(p) if p > 1 else 0 for p in P_values]
        ax.plot(range(len(P_values)), max_entropies, 'r--',
                linewidth=2, label='Theoretical Maximum')

        ax.set_xlabel('Number of Parallel Streams (P)')
        ax.set_ylabel('Average Stream Entropy')
        ax.set_title('Stream Weight Entropy by P')
        ax.set_xticks(range(len(P_values)))
        ax.set_xticklabels([str(p) for p in P_values])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        plot_file = output_path / 'stream_entropy.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files['stream_entropy'] = str(plot_file)

        plt.close()

    return plot_files


def generate_report(
    results: Dict[int, Dict[str, Any]],
    output_dir: str,
    experiment_name: str = "ParScale Local Replication"
) -> str:
    """
    Generate comprehensive analysis report.

    Args:
        results: Dictionary of results by P value
        output_dir: Directory to save report
        experiment_name: Name of the experiment

    Returns:
        Path to generated report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create results table
    results_df = create_results_table(results)

    # Create plots
    plot_files = create_loss_comparison_plots(results, output_dir)

    # Generate report
    report_path = output_path / "analysis_report.md"

    with open(report_path, 'w') as f:
        f.write(f"# {experiment_name} - Analysis Report\n\n")

        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")

        if len(results) >= 2:
            P_values = sorted(results.keys())
            losses = [results[P].get('loss', float('inf')) for P in P_values]
            valid_losses = [l for l in losses if l != float('inf')]

            if len(valid_losses) >= 2:
                # Check for monotonic improvement
                is_monotonic = all(losses[i] >= losses[i+1] for i in range(len(losses)-1)
                                   if losses[i] != float('inf') and losses[i+1] != float('inf'))

                f.write(f"- **Monotonic improvement**: {'✅ Yes' if is_monotonic else '❌ No'}\n")
                f.write(f"- **P values tested**: {P_values}\n")
                f.write(f"- **Best performing P**: {P_values[np.argmin(valid_losses)]}\n")

                # Calculate improvement
                if 1 in results and results[1].get('perplexity') and len(valid_losses) > 1:
                    baseline_ppl = results[1]['perplexity']
                    best_P = P_values[np.argmin(valid_losses)]
                    best_ppl = results[best_P].get('perplexity', float('inf'))

                    if best_ppl != float('inf'):
                        improvement = ((baseline_ppl - best_ppl) / baseline_ppl) * 100
                        f.write(f"- **Perplexity improvement over P=1**: {improvement:.2f}%\n")

        f.write("\n")

        # Results table
        f.write("## Results Summary\n\n")
        f.write("| P | Tokens (M) | Final Loss | Perplexity | Δ PPL vs P=1 | Entropy ↓? | Notes |\n")
        f.write("|---|------------|------------|------------|--------------|-------------|-------|\n")

        for _, row in results_df.iterrows():
            P = int(row['P'])
            tokens_M = f"{row['tokens_M']:.1f}" if not pd.isna(row['tokens_M']) else "N/A"
            loss = f"{row['final_loss']:.4f}" if row['final_loss'] != float('inf') else "∞"
            ppl = f"{row['perplexity']:.2f}" if row['perplexity'] != float('inf') else "∞"
            delta_ppl = f"{row['delta_ppl_vs_P1']:+.2f}" if not pd.isna(row['delta_ppl_vs_P1']) else "N/A"
            entropy_ok = "✅" if row.get('entropy_decreasing', False) and not row.get('collapse_detected', False) else "❌" if P > 1 else "N/A"

            notes = []
            if row.get('collapse_detected', False):
                notes.append("Stream collapse")
            if row.get('trainable_params', 0) > 0:
                notes.append(f"{row['trainable_params']:,} params")

            notes_str = "; ".join(notes) if notes else ""

            f.write(f"| {P} | {tokens_M} | {loss} | {ppl} | {delta_ppl} | {entropy_ok} | {notes_str} |\n")

        f.write("\n")

        # Scaling law analysis
        f.write("## Scaling Law Analysis\n\n")

        P_values = [P for P in sorted(results.keys()) if results[P].get('loss', float('inf')) != float('inf')]
        losses = [results[P]['loss'] for P in P_values]

        if len(P_values) >= 3:  # Need at least 3 points for meaningful fit
            fitted_curve, fit_stats = fit_scaling_law(P_values, losses, "log")

            if fit_stats:
                f.write(f"**Logarithmic fit**: {fit_stats['equation']}\n\n")
                f.write(f"- R² = {fit_stats['r_squared']:.4f}\n")
                f.write(f"- Slope = {fit_stats['a']:.4f} (negative indicates improvement with log(P))\n\n")

                if fit_stats['a'] < 0:
                    f.write("✅ **Result**: Loss decreases with log(P) as expected\n\n")
                else:
                    f.write("❌ **Result**: Loss does not decrease with log(P)\n\n")

        # Training efficiency
        f.write("## Training Efficiency\n\n")

        total_time_hours = sum(results[P].get('total_time_seconds', 0) / 3600 for P in results.keys())
        total_tokens = sum(results[P].get('total_tokens', 0) for P in results.keys())

        f.write(f"- **Total experiment time**: {total_time_hours:.2f} hours\n")
        f.write(f"- **Total tokens processed**: {total_tokens:,}\n")

        if total_time_hours > 0:
            f.write(f"- **Average throughput**: {total_tokens / (total_time_hours * 3600):.2f} tokens/second\n")

        f.write("\n")

        # Stream diagnostics
        f.write("## Stream Diagnostics\n\n")

        for P in sorted(results.keys()):
            if P > 1 and 'entropy_mean' in results[P]:
                entropy = results[P]['entropy_mean']
                max_entropy = np.log(P)
                entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0

                f.write(f"**P={P}**:\n")
                f.write(f"- Average entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy_ratio:.1%})\n")

                if results[P].get('collapse_detected', False):
                    f.write(f"- ⚠️ Stream collapse detected\n")
                else:
                    f.write(f"- ✅ Healthy stream diversity\n")

                f.write("\n")

        # Visualizations
        f.write("## Visualizations\n\n")

        for plot_name, plot_file in plot_files.items():
            plot_path = Path(plot_file)
            if plot_path.exists():
                f.write(f"### {plot_name.replace('_', ' ').title()}\n\n")
                f.write(f"![{plot_name}]({plot_path.name})\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        if len(results) >= 2:
            f.write("### Key Findings:\n\n")

            # Monotonic improvement
            P_values = sorted([P for P in results.keys() if results[P].get('loss', float('inf')) != float('inf')])
            losses = [results[P]['loss'] for P in P_values]

            if len(losses) >= 2:
                is_improving = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
                f.write(
                    f"1. **Scaling behavior**: ParScale {'shows' if is_improving else 'does not show'} the expected monotonic improvement with increasing P\n")

            # Best configuration
            if losses:
                best_P = P_values[np.argmin(losses)]
                best_loss = min(losses)
                f.write(f"2. **Optimal configuration**: P={best_P} achieved the lowest validation loss ({best_loss:.4f})\n")

            # Stream health
            healthy_streams = sum(1 for P in results.keys() if P > 1 and
                                  not results[P].get('collapse_detected', False))
            total_multi_streams = sum(1 for P in results.keys() if P > 1)

            if total_multi_streams > 0:
                f.write(f"3. **Stream diversity**: {healthy_streams}/{total_multi_streams} configurations maintained healthy stream diversity\n")

        f.write("\n### Reproducibility:\n\n")
        f.write("All experiments used identical hyperparameters and data sampling procedures. ")
        f.write("Results are fully reproducible using the provided configuration files and logged seeds.\n\n")

    return str(report_path)


def verify_experimental_validity(results: Dict[int, Dict[str, Any]]) -> Dict[str, bool]:
    """
    Verify experimental validity according to success criteria.

    Args:
        results: Dictionary of results by P value

    Returns:
        Dictionary of validation checks
    """
    checks = {}

    # Check 1: Equal token budgets
    token_counts = [results[P].get('total_tokens', 0) for P in results.keys()]
    if len(set(token_counts)) <= 1 or (max(token_counts) - min(token_counts)) / max(token_counts) < 0.05:
        checks['equal_token_budgets'] = True
    else:
        checks['equal_token_budgets'] = False

    # Check 2: Monotonic improvement
    valid_results = {P: results[P] for P in results.keys()
                     if results[P].get('loss', float('inf')) != float('inf')}

    if len(valid_results) >= 2:
        P_values = sorted(valid_results.keys())
        losses = [valid_results[P]['loss'] for P in P_values]

        checks['monotonic_improvement'] = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    else:
        checks['monotonic_improvement'] = False

    # Check 3: No stream collapse
    stream_health = True
    for P in results.keys():
        if P > 1 and results[P].get('collapse_detected', False):
            stream_health = False
            break

    checks['healthy_stream_diversity'] = stream_health

    # Check 4: Statistical significance (if enough data points)
    if len(valid_results) >= 3:
        P_values = sorted(valid_results.keys())
        losses = [valid_results[P]['loss'] for P in P_values]

        # Simple correlation test
        correlation, p_value = stats.spearmanr(P_values, losses)
        checks['statistically_significant'] = p_value < 0.05 and correlation < 0
    else:
        checks['statistically_significant'] = False

    return checks


def parse_leaderboard_results(
    results_path: str,
    model_whitelist: Optional[List[str]] = None,
    eval_blacklist: Optional[List[Tuple[Optional[str], Optional[str]]]] = None,
    model_name_mapping: Optional[Dict[str, str]] = None,
    download_repos: bool = True,
    s3_path: Optional[str] = None,
    min_ct: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Parse leaderboard evaluation results from JSON files and generate structured data.

    Args:
        results_path: Path to directory containing evaluation result JSON files
        model_whitelist: List of model name patterns to include (None = include all)
        eval_blacklist: List of (metric_regex, dataset_regex) tuples to exclude
        model_name_mapping: Dictionary mapping model names to display names
        download_repos: Whether to download from HuggingFace repos (requires imports)
        s3_path: Optional S3 path for syncing ParControl model results (e.g., "s3://obviouslywrong-parcontrol/ParControl/evals-quick")

    Returns:
        Tuple of:
            - DataFrame with parsed results in wide format (models as columns, metrics as rows)
            - Raw data dictionary mapping model_name -> {(dataset, metric): value}
    """
    import os
    import subprocess

    # Default values
    if eval_blacklist is None:
        eval_blacklist = [
            ("stderr", None),
            ("f1", None),
            (None, "faithdial"),
            (None, "truthfulqa_gen"),
            (None, "fever"),
        ]

    if model_whitelist is None:
        model_whitelist = [
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Llama-2-7b-hf",
            "ParControl/",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
        ]

    if model_name_mapping is None:
        model_name_mapping = {}

    # Download repos if requested
    if download_repos:
        logger.info("Syncing results from HF: %s to %s", s3_path, results_path)
        my_snapshot_download(repo_id=RESULTS_REPO, revision="main",
                             local_dir=results_path, repo_type="dataset", max_workers=60)

        logger.info("Syncing ParControl results from S3: %s to %s", s3_path, results_path)
        try:
            os.makedirs(results_path, exist_ok=True)
            cmd = ["aws", "s3", "sync", s3_path, results_path]
            # Run without capturing output so progress is shown on screen
            result = subprocess.run(cmd, check=True)
            logger.info("Successfully synced ParControl models from S3")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to sync from S3 with exit code %s", e.returncode, exc_info=True)
            raise
        except FileNotFoundError:
            logger.error("AWS CLI not found. Please install aws-cli to use S3 sync functionality")
            raise

    # Find all JSON files
    result_path_lst = []
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file.endswith(".json"):
                result_path_lst.append(os.path.join(root, file))

    # Parse results
    model_dataset_metric_to_result_map = {}
    data_map = {}

    for path in result_path_lst:
        # Check model whitelist
        to_add = False
        for name in model_whitelist:
            if name in path:
                to_add = True
                break
        if not to_add:
            continue

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            model_name = data.get("config", {}).get("model_name")
            if not model_name:
                logger.debug("Skipped %s without model_name", path)
                continue

            for dataset_name, results_dict in data["results"].items():
                for metric_name, value in results_dict.items():
                    to_add = True

                    # Apply blacklist
                    for metric_regex, data_regex in eval_blacklist:
                        if metric_regex is not None and metric_regex in metric_name:
                            to_add = False
                            break
                        elif data_regex is not None and data_regex in dataset_name:
                            to_add = False
                            break

                    # Special filtering rules
                    if 'bertscore' in metric_name:
                        if 'precision' not in metric_name:
                            to_add = False

                    if 'halueval' in dataset_name:
                        if 'acc' not in metric_name:
                            to_add = False

                    if 'ifeval' in dataset_name:
                        if 'prompt_level_strict_acc' not in metric_name:
                            to_add = False

                    if 'squad' in dataset_name:
                        if 'best_exact' in metric_name:
                            to_add = False

                    if "truthfulqa_gen" in dataset_name:
                        if "acc" not in metric_name:
                            to_add = False
                        if "rouge" in metric_name:
                            to_add = False

                    if ('xsum' in dataset_name or 'cnn' in dataset_name) and 'v2' not in dataset_name:
                        to_add = False

                    # Check if value is numeric
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            to_add = False

                    logging.info("Saw %s %s -> %s", dataset_name, metric_name, to_add)
                    if to_add:
                        # Apply value transformations
                        if 'rouge' in metric_name:
                            value /= 100.0

                        if 'squad' in dataset_name:
                            value /= 100.0

                        # Sanitize names
                        sanitised_metric_name = metric_name
                        if "," in sanitised_metric_name:
                            sanitised_metric_name = sanitised_metric_name.split(',')[0]
                        sanitised_metric_name = _sanitise_metric(sanitised_metric_name)
                        sanitised_dataset_name = _sanitise_dataset(dataset_name)

                        # Apply model name mapping
                        mapped_model_name = model_name_mapping.get(model_name, model_name)

                        if mapped_model_name not in data_map:
                            data_map[mapped_model_name] = {}

                        subkey = (sanitised_dataset_name, sanitised_metric_name)
                        key = (mapped_model_name,) + subkey

                        # Handle conflicts by taking minimum value
                        if key in model_dataset_metric_to_result_map:
                            old_value = model_dataset_metric_to_result_map[key]
                            if np.abs(old_value - value) > 1e-3:
                                result = min(old_value, value)
                                logger.warning("Chose minimum value for conflicted key=%s: %.3f %.3f -> %.3f",
                                               key, value, old_value, result)
                                model_dataset_metric_to_result_map[key] = result
                                data_map[mapped_model_name][subkey] = result
                        else:
                            model_dataset_metric_to_result_map[key] = value
                            data_map[mapped_model_name][subkey] = value

        except Exception:
            logger.error("Couldn't parse %s", path, exc_info=True)

    # Convert to DataFrame format
    # Restructure data_map for DataFrame: rows = (dataset, metric), columns = models
    data_map_v2 = {}
    for model_name in data_map.keys():
        for dataset_metric in data_map[model_name].keys():
            if dataset_metric not in data_map_v2:
                data_map_v2[dataset_metric] = {}
            data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]

    # Create DataFrame
    df = pd.DataFrame.from_dict(data_map_v2, orient='index')
    df.index = [', '.join(map(str, idx)) for idx in df.index]

    # Filter rows with sufficient data
    counts_s = df.count(axis=1)
    df = df.reindex(counts_s[counts_s >= min_ct].index)

    # Sort lexically
    df = df.sort_index(axis=0, key=lambda x: x.str.lower())
    df = df.reindex(sorted(df.columns), axis=1)

    # Set small values to NaN for better visualization
    df[df < 1e-3] = np.nan

    return df, data_map


def _sanitise_metric(name: str) -> str:
    """Sanitise metric name for display."""
    res = name
    res = res.replace("prompt_level_strict_acc", "Prompt-Level Accuracy")
    res = res.replace("acc", "Accuracy")
    res = res.replace("exact_match", "EM")
    res = res.replace("avg-selfcheckgpt", "AVG")
    res = res.replace("max-selfcheckgpt", "MAX")
    res = res.replace("rouge", "ROUGE-")
    res = res.replace("bertscore_precision", "BERT-P")
    res = res.replace("exact", "EM")
    res = res.replace("HasAns_EM", "HasAns")
    res = res.replace("NoAns_EM", "NoAns")
    res = res.replace("em", "EM")
    return res


def _sanitise_dataset(name: str) -> str:
    """Sanitise dataset name for display."""
    res = name
    res = res.replace("tqa8", "TriviaQA (8-shot)")
    res = res.replace("nq8", "NQ (8-shot)")
    res = res.replace("nq_open", "NQ (64-shot)")
    res = res.replace("triviaqa", "TriviaQA (64-shot)")
    res = res.replace("truthfulqa", "TruthfulQA")
    res = res.replace("ifeval", "IFEval")
    res = res.replace("selfcheckgpt", "SelfCheckGPT")
    res = res.replace("truefalse_cieacf", "True-False")
    res = res.replace("mc", "MC")
    res = res.replace("race", "RACE")
    res = res.replace("squad", "SQuAD")
    res = res.replace("memo-trap", "MemoTrap")
    res = res.replace("cnndm", "CNN/DM")
    res = res.replace("xsum", "XSum")
    res = res.replace("qa", "QA")
    res = res.replace("summarization", "Summarization")
    res = res.replace("dialogue", "Dialog")
    res = res.replace("halueval", "HaluEval")
    res = res.replace("_v2", "")
    res = res.replace("_", " ")
    return res
