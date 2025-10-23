"""
Stream diagnostics for ParScale experiments.
Analyzes stream weights, usage patterns, and diversity metrics.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from peft import PeftModel
from scipy.stats import entropy


class StreamDiagnostics:
    """
    Comprehensive stream diagnostics for ParScale models.
    """

    def __init__(self, P: int):
        self.P = P
        self.history = []

    def extract_stream_weights(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract per-token stream weights from ParScale model.

        Args:
            model: ParScale model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Stream weights tensor [batch_size, seq_len, P]
        """
        model.eval()
        with torch.no_grad():
            # Forward pass to get hidden states
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            if self.P == 1:
                # For P=1, create uniform weights
                batch_size, seq_len = input_ids.shape
                weights = torch.ones(batch_size, seq_len, 1, device=input_ids.device)
                return weights

            # Get final hidden states before aggregation
            hidden_states = outputs.hidden_states  # [batch_size, seq_len, hidden_size]

            if isinstance(model, PeftModel):
                model = model.base_model.model

            # Get stream weights from aggregation layer
            if hasattr(model.model, 'aggregate_layer'):
                # Reshape for aggregation layer input
                reshaped = rearrange(hidden_states, "(n_parscale b) s h -> b s (h n_parscale)", n_parscale=self.P)
                logits = model.model.aggregate_layer(reshaped)  # [batch_size, seq_len, P]
                weights = torch.softmax(logits.float(), dim=-1)

                # Apply smoothing if configured
                if hasattr(model.model, 'parscale_aggregate_attn_smoothing'):
                    smooth = model.model.parscale_aggregate_attn_smoothing
                    weights = weights * (1 - smooth) + (smooth / self.P)

                return weights
            else:
                # Fallback: uniform weights
                batch_size, seq_len = input_ids.shape[:2]
                weights = torch.ones(batch_size, seq_len, self.P, device=input_ids.device) / self.P
                return weights

    def compute_stream_statistics(
        self,
        weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive stream statistics.

        Args:
            weights: Stream weights [batch_size, seq_len, P]
            attention_mask: Mask for valid positions [batch_size, seq_len]

        Returns:
            Dictionary of statistics
        """
        # Convert to numpy for easier computation
        weights_np = weights.detach().cpu().numpy()

        if attention_mask is not None:
            mask_np = attention_mask.detach().cpu().numpy()
            # Apply mask: set weights to uniform for masked positions
            for i in range(weights_np.shape[0]):
                for j in range(weights_np.shape[1]):
                    if mask_np[i, j] == 0:
                        weights_np[i, j, :] = 1.0 / self.P

        # Flatten to [total_valid_positions, P]
        if attention_mask is not None:
            valid_positions = []
            for i in range(weights_np.shape[0]):
                for j in range(weights_np.shape[1]):
                    if mask_np[i, j] == 1:
                        valid_positions.append(weights_np[i, j, :])
            if valid_positions:
                weights_flat = np.array(valid_positions)
            else:
                weights_flat = weights_np.reshape(-1, self.P)
        else:
            weights_flat = weights_np.reshape(-1, self.P)

        if len(weights_flat) == 0:
            return self._empty_statistics()

        # Compute statistics
        stats = {}

        # Entropy statistics
        eps = 1e-8
        entropies = -np.sum(weights_flat * np.log(weights_flat + eps), axis=1)
        stats['entropy_mean'] = float(np.mean(entropies))
        stats['entropy_std'] = float(np.std(entropies))
        stats['entropy_min'] = float(np.min(entropies))
        stats['entropy_max'] = float(np.max(entropies))
        stats['entropy_p25'] = float(np.percentile(entropies, 25))
        stats['entropy_p50'] = float(np.percentile(entropies, 50))
        stats['entropy_p75'] = float(np.percentile(entropies, 75))
        stats['entropy_p95'] = float(np.percentile(entropies, 95))

        # Argmax statistics
        argmax_streams = np.argmax(weights_flat, axis=1)
        for i in range(self.P):
            count = np.sum(argmax_streams == i)
            stats[f'argmax_rate_{i}'] = float(count / len(argmax_streams))

        # Mean weight per stream
        mean_weights = np.mean(weights_flat, axis=0)
        for i in range(self.P):
            stats[f'mean_weight_{i}'] = float(mean_weights[i])

        # KL divergence from uniform distribution
        uniform = np.ones(self.P) / self.P
        kl_divergences = []
        for w in weights_flat:
            kl = np.sum(w * np.log((w + eps) / (uniform + eps)))
            kl_divergences.append(kl)
        stats['kl_to_uniform_mean'] = float(np.mean(kl_divergences))
        stats['kl_to_uniform_std'] = float(np.std(kl_divergences))

        # Diversity metrics
        stats['max_weight_mean'] = float(np.mean(np.max(weights_flat, axis=1)))
        stats['min_weight_mean'] = float(np.mean(np.min(weights_flat, axis=1)))
        stats['weight_range_mean'] = stats['max_weight_mean'] - stats['min_weight_mean']

        # Gini coefficient (measure of inequality)
        gini_coeffs = []
        for w in weights_flat:
            sorted_w = np.sort(w)
            n = len(sorted_w)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_w)) / (n * np.sum(sorted_w)) - (n + 1) / n
            gini_coeffs.append(gini)
        stats['gini_coefficient_mean'] = float(np.mean(gini_coeffs))

        return stats

    def _empty_statistics(self) -> Dict[str, float]:
        """Return empty/default statistics when no valid positions."""
        stats = {
            'entropy_mean': 0.0, 'entropy_std': 0.0, 'entropy_min': 0.0, 'entropy_max': 0.0,
            'entropy_p25': 0.0, 'entropy_p50': 0.0, 'entropy_p75': 0.0, 'entropy_p95': 0.0,
            'kl_to_uniform_mean': 0.0, 'kl_to_uniform_std': 0.0,
            'max_weight_mean': 1.0/self.P, 'min_weight_mean': 1.0/self.P, 'weight_range_mean': 0.0,
            'gini_coefficient_mean': 0.0
        }

        # Add per-stream statistics
        for i in range(self.P):
            stats[f'argmax_rate_{i}'] = 1.0/self.P
            stats[f'mean_weight_{i}'] = 1.0/self.P

        return stats

    def record_statistics(self, step: int, stats: Dict[str, float]):
        """Record statistics for historical tracking."""
        record = {'step': step, **stats}
        self.history.append(record)

    def detect_stream_collapse(self, stats: Dict[str, float], threshold: float = 0.1) -> bool:
        """
        Detect if streams have collapsed (low diversity).

        Args:
            stats: Stream statistics
            threshold: Entropy threshold below which collapse is detected

        Returns:
            True if collapse detected
        """
        return stats['entropy_mean'] < threshold

    def create_visualization(
        self,
        weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive stream diagnostics visualization.

        Args:
            weights: Stream weights [batch_size, seq_len, P]
            attention_mask: Attention mask
            step: Current training step
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        weights_np = weights.detach().cpu().numpy()

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_np = attention_mask.detach().cpu().numpy()
            for i in range(weights_np.shape[0]):
                for j in range(weights_np.shape[1]):
                    if mask_np[i, j] == 0:
                        weights_np[i, j, :] = np.nan

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Stream Diagnostics - Step {step} (P={self.P})', fontsize=16)

        # 1. Stream weights heatmap (first batch, first few sequences)
        sample_weights = weights_np[0]  # [seq_len, P]
        im1 = axes[0, 0].imshow(sample_weights.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title('Stream Weights Over Sequence (Sample)')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Stream ID')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Average stream weights across all samples
        mean_weights = np.nanmean(weights_np, axis=0)  # [seq_len, P]
        im2 = axes[0, 1].imshow(mean_weights.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('Average Stream Weights')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Stream ID')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Entropy distribution
        eps = 1e-8
        entropies = -np.nansum(weights_np * np.log(weights_np + eps), axis=-1)
        valid_entropies = entropies[~np.isnan(entropies)]
        if len(valid_entropies) > 0:
            axes[0, 2].hist(valid_entropies.flatten(), bins=50, alpha=0.7, density=True)
            axes[0, 2].axvline(np.log(self.P), color='red', linestyle='--', label=f'Max Entropy (log({self.P}))')
            axes[0, 2].set_title('Entropy Distribution')
            axes[0, 2].set_xlabel('Entropy')
            axes[0, 2].set_ylabel('Density')
            axes[0, 2].legend()

        # 4. Argmax stream frequency
        valid_weights = weights_np[~np.isnan(weights_np).any(axis=2)]
        if len(valid_weights) > 0:
            argmax_streams = np.argmax(valid_weights, axis=1)
            unique, counts = np.unique(argmax_streams, return_counts=True)
            axes[1, 0].bar(unique, counts / len(argmax_streams))
            axes[1, 0].set_title('Stream Selection Frequency (Argmax)')
            axes[1, 0].set_xlabel('Stream ID')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_xticks(range(self.P))

        # 5. Mean weight per stream
        mean_per_stream = np.nanmean(weights_np.reshape(-1, self.P), axis=0)
        axes[1, 1].bar(range(self.P), mean_per_stream)
        axes[1, 1].axhline(1.0/self.P, color='red', linestyle='--', label='Uniform')
        axes[1, 1].set_title('Mean Weight Per Stream')
        axes[1, 1].set_xlabel('Stream ID')
        axes[1, 1].set_ylabel('Mean Weight')
        axes[1, 1].legend()

        # 6. Weight distribution per stream
        weights_flat = weights_np.reshape(-1, self.P)
        valid_mask = ~np.isnan(weights_flat).any(axis=1)
        weights_valid = weights_flat[valid_mask]

        if len(weights_valid) > 0:
            for i in range(min(self.P, 8)):  # Show up to 8 streams
                axes[1, 2].hist(weights_valid[:, i], bins=30, alpha=0.5, label=f'Stream {i}', density=True)
            axes[1, 2].set_title('Weight Distribution by Stream')
            axes[1, 2].set_xlabel('Weight')
            axes[1, 2].set_ylabel('Density')
            if self.P <= 8:
                axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def create_entropy_evolution_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create plot showing entropy evolution over training steps.

        Args:
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not self.history:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No history available', ha='center', va='center', transform=ax.transAxes)
            return fig

        df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Stream Evolution Over Training (P={self.P})', fontsize=16)

        # Entropy evolution
        axes[0, 0].plot(df['step'], df['entropy_mean'], label='Mean', linewidth=2)
        axes[0, 0].fill_between(df['step'],
                                df['entropy_mean'] - df['entropy_std'],
                                df['entropy_mean'] + df['entropy_std'],
                                alpha=0.3)
        axes[0, 0].axhline(np.log(self.P), color='red', linestyle='--', label=f'Max Entropy (log({self.P}))')
        axes[0, 0].set_title('Entropy Evolution')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Argmax rates evolution
        for i in range(self.P):
            if f'argmax_rate_{i}' in df.columns:
                axes[0, 1].plot(df['step'], df[f'argmax_rate_{i}'], label=f'Stream {i}')
        axes[0, 1].axhline(1.0/self.P, color='red', linestyle='--', label='Uniform')
        axes[0, 1].set_title('Argmax Selection Rates')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Selection Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Mean weights evolution
        for i in range(self.P):
            if f'mean_weight_{i}' in df.columns:
                axes[1, 0].plot(df['step'], df[f'mean_weight_{i}'], label=f'Stream {i}')
        axes[1, 0].axhline(1.0/self.P, color='red', linestyle='--', label='Uniform')
        axes[1, 0].set_title('Mean Weights Evolution')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Mean Weight')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Diversity metrics
        axes[1, 1].plot(df['step'], df['gini_coefficient_mean'], label='Gini Coefficient', linewidth=2)
        axes[1, 1].plot(df['step'], df['weight_range_mean'], label='Weight Range', linewidth=2)
        axes[1, 1].set_title('Diversity Metrics')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all recorded history."""
        if not self.history:
            return {}

        df = pd.DataFrame(self.history)

        summary = {
            'total_steps': len(df),
            'final_entropy_mean': float(df['entropy_mean'].iloc[-1]),
            'entropy_trend': float(np.polyfit(df['step'], df['entropy_mean'], 1)[0]),  # Linear slope
            'min_entropy_reached': float(df['entropy_mean'].min()),
            'max_entropy_reached': float(df['entropy_mean'].max()),
        }

        # Check for collapse
        collapse_threshold = 0.1
        collapse_steps = df[df['entropy_mean'] < collapse_threshold]
        summary['collapse_detected'] = len(collapse_steps) > 0
        summary['first_collapse_step'] = int(collapse_steps['step'].iloc[0]) if len(collapse_steps) > 0 else None

        # Stream balance
        final_argmax_rates = [df[f'argmax_rate_{i}'].iloc[-1] for i in range(self.P) if f'argmax_rate_{i}' in df.columns]
        if final_argmax_rates:
            summary['final_stream_balance'] = float(np.std(final_argmax_rates))
            summary['most_used_stream'] = int(np.argmax(final_argmax_rates))
            summary['least_used_stream'] = int(np.argmin(final_argmax_rates))

        return summary
