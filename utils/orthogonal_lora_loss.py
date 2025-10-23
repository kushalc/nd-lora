"""
Orthogonal LoRA loss implementation for L-1 experiment.
Includes running mean normalization and composite loss with Barlow Twins,
orthogonality penalty, and knowledge distillation.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.stream_aware_lora import parse_streams_from_batch

logger = logging.getLogger(__name__)


class RunningMeanNormalizer:
    """Normalizes loss components to target value using exponential moving average."""

    def __init__(self, target: float = 5.0, alpha: float = 0.1, eps: float = 1e-5):
        self.target = target
        self.running_mean = None
        self.count = 0
        self.alpha = alpha
        self.eps = eps

    def update_and_normalize(self, value: torch.Tensor) -> torch.Tensor:
        """Update running mean and normalize (used during warmup)."""
        # Don't require gradients for normalization - we'll preserve the gradient status
        if not value.requires_grad:
            return value * 0.0  # Return zero if no gradients

        with torch.no_grad():
            if self.running_mean is None:
                self.running_mean = value.detach().clone()
            else:
                self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * value.detach()

        self.count += 1
        return (value / (self.running_mean + self.eps)) * self.target

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize using fixed running mean (used after warmup)."""
        if not value.requires_grad or self.running_mean is None:
            return value * 0.0  # Return zero if no gradients or not initialized
        return (value / (self.running_mean + self.eps)) * self.target


class OrthogonalLoRALoss(nn.Module):
    """
    Composite loss for L-1 experiment combining CE, Barlow Twins,
    orthogonality penalty, and knowledge distillation.
    """

    def __init__(
        self,
        P: int,
        warmup_steps: int,
        design_layer: int,
        lambda_bt: float = 0.1,
        lambda_perp: float = 0.5,
        lambda_kd: float = 0.05,
        bt_method: str = "mean_vs_others",
        bt_k: int = None,
        bt_normalization_warmup: bool = False,
    ):
        super().__init__()
        self.P = P
        self.warmup_steps = warmup_steps
        self.design_layer = design_layer
        self.lambda_bt = lambda_bt
        self.lambda_perp = lambda_perp
        self.lambda_kd = lambda_kd
        self.bt_method = bt_method
        self.bt_k = bt_k if bt_k else P
        self.bt_normalization_warmup = bt_normalization_warmup

        if self.bt_normalization_warmup:
            self.normalizer = RunningMeanNormalizer(target=20)  # NOTE: During warmup
        else:
            self.normalizer = None

        # Normalizers for each component
        # self.bt_normalizer = RunningMeanNormalizer(target=5.0)
        # self.orth_normalizer = RunningMeanNormalizer(target=5.0)
        # self.kd_normalizer = RunningMeanNormalizer(target=5.0)

    def forward(
        self,
        step: int,
        model,
        hidden_states: torch.Tensor,
        logits_agg: torch.Tensor,
        logits_backbone: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute L-1 composite loss.

        Args:
            step: Current training step
            model: PEFT model with LoRA
            hidden_states_by_stream: Hidden states [P, batch, seq, hidden]
            logits_agg: ParScale aggregated logits
            logits_backbone: Backbone model logits

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        hidden_states_by_stream = parse_streams_from_batch(hidden_states, self.P)

        # Choose Barlow Twins method based on configuration
        if self.bt_method == "mean_vs_others":
            bt_loss = self._compute_barlow_twins_with_mean_vs_others(hidden_states_by_stream)
        elif self.bt_method == "standard":
            bt_loss = self._compute_standard_barlow_twins(hidden_states_by_stream)
        else:
            raise ValueError(f"Unknown bt_method: {self.bt_method}. Must be 'mean_vs_others' or 'standard'")

        # orth_loss = self._compute_orthogonality_penalty(model)
        # kd_loss = self._compute_kd_loss(logits_agg, logits_backbone)

        warmup = 1.0
        normalized_loss = bt_loss
        if self.bt_normalization_warmup:
            warmup = min(step / self.warmup_steps + 1e-3, 1)
            if step <= self.warmup_steps:
                normalized_loss = self.normalizer.update_and_normalize(bt_loss)
            else:
                normalized_loss = self.normalizer.normalize(bt_loss)

        # Combine with weights
        total_loss = self.lambda_bt * warmup * normalized_loss
        loss_components = {
            "loss/bt": normalized_loss,
            "loss/bt_raw": bt_loss,
            # "loss/orth": orth_loss,
            "loss/total": total_loss
        }

        return total_loss, loss_components

    def _compute_global_variance_scale(self,
                                       hidden_states_by_stream: torch.Tensor,
                                       alpha: float = 0.5,
                                       eps: float = 1e-5) -> torch.Tensor:
        """
        Compute global variance scaling factor across streams.

        Args:
            hidden_states_by_stream: [P, batch, seq, hidden]
            alpha: Inverse-variance scaling exponent
            eps: Small constant for numerical stability

        Returns:
            Scaling factor (detached)
        """
        P, B, S, D = hidden_states_by_stream.shape
        N = B * S

        # Flatten and standardize per stream
        Z = hidden_states_by_stream.view(P, N, D)
        Z = (Z - Z.mean(1, keepdim=True)) / (Z.std(1, keepdim=True) + eps)

        # Global variance across streams
        var_all = Z.var(dim=0, unbiased=False).mean()
        scale = (var_all + eps).pow(alpha).detach()

        return scale

    def _compute_standard_barlow_twins(self, hidden_states_by_stream: torch.Tensor) -> torch.Tensor:
        """Sample K pairs per stream proportionally to cross-correlation for BT loss."""
        P, batch_size, seqlen, hidden_size = hidden_states_by_stream.shape
        device = hidden_states_by_stream.device
        assert P >= 2, f"Need at least 2 streams for Barlow Twins, got {P}"

        # Normalize representations
        reps_norm = []
        for p in range(P):
            rep = hidden_states_by_stream[p].view(-1, hidden_states_by_stream.size(-1))
            reps_norm.append((rep - rep.mean(0)) / (rep.std(0) + 1e-8))

        # Sample K pairs per stream
        sampled = []
        I = torch.eye(reps_norm[0].size(-1), device=device)
        for p in range(P):
            losses = []
            for q in range(p + 1, P):
                C = torch.mm(reps_norm[p].T, reps_norm[q]) / reps_norm[0].size(0)
                losses.append(torch.norm(C - I, p='fro'))

            if not losses:
                continue

            selected = range(len(losses))
            if self.bt_k < len(losses):
                k = self.bt_k
                if self.bt_k < 0:
                    k = max(len(losses) + self.bt_k, 1)
                probs = torch.tensor([loss.detach() for loss in losses], device=device)
                selected = torch.multinomial(probs, k, replacement=False)
            sampled += [losses[j] for j in selected]

        hidden_factor = hidden_size * (hidden_size - 1) / 2
        bt_loss = sum(sampled) / len(sampled) / hidden_factor * 4096

        return bt_loss

    def _compute_barlow_twins_with_mean_vs_others(self,
                                                  hidden_states_by_stream: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-of-others Barlow Twins loss: (1/P) Σ_p ||C(p, mean_{q≠p}) - I||_F^2

        Args:
            hidden_states_by_stream: [P, batch, seq, hidden]

        Returns:
            Barlow Twins loss tensor
        """
        P, batch_size, seqlen, hidden_size = hidden_states_by_stream.shape
        assert P >= 2, f"Need at least 2 streams for Barlow Twins, got {P}"

        # Extract representations at design layer (already extracted per stream)
        reps = []
        for p in range(P):
            rep = hidden_states_by_stream[p]  # (batch, seq, hidden)
            rep_flat = rep.view(-1, rep.size(-1))  # (batch*seq, hidden)
            reps.append(rep_flat)

        # Stack all representations: [P, N, D]
        Z = torch.stack(reps, dim=0)

        # Per-stream standardization (zero mean, unit variance)
        Z_norm = (Z - Z.mean(dim=1, keepdim=True)) / (Z.std(dim=1, keepdim=True) + 1e-8)

        # Precompute sum over streams for efficient mean-of-others
        sum_Z = Z_norm.sum(dim=0, keepdim=True)  # [1, N, D]

        bt_loss = torch.tensor(0.0, device=hidden_states_by_stream.device, requires_grad=True)
        for p in range(P):
            rep_p = Z_norm[p]  # (N, D)
            # Mean of all other streams
            rep_mean_others = (sum_Z[0] - rep_p) / (P - 1)  # (N, D)

            # Cross-correlation matrix
            N = rep_p.size(0)
            C = torch.mm(rep_p.T, rep_mean_others) / N  # (D, D)

            # Identity matrix
            I = torch.eye(C.size(0), device=C.device, dtype=C.dtype)

            # Full Barlow Twins loss (C - I)
            bt_loss = bt_loss + torch.norm(C - I, p='fro') ** 2

        # Normalize by hidden size & # of stream combinations
        stream_count = hidden_states_by_stream.shape[0]
        stream_combinations = stream_count * (stream_count - 1) / 2
        hidden_factor = hidden_size * (hidden_size - 1) / 2
        scaled_loss = bt_loss / stream_combinations / hidden_factor

        # Normalize by inverse variance
        variance_scale = self._compute_global_variance_scale(hidden_states_by_stream)
        normed_loss = scaled_loss / variance_scale

        return normed_loss

    def _compute_kd_loss(self, logits_agg: torch.Tensor, logits_backbone: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence: KL(softmax(z_agg) || softmax(z_backbone))

        Args:
            logits_agg: ParScale aggregated logits
            logits_backbone: Backbone model logits

        Returns:
            KL divergence loss
        """
        # Reshape to (batch*seq, vocab) if needed
        if logits_agg.dim() > 2:
            logits_agg = logits_agg.view(-1, logits_agg.size(-1))
        if logits_backbone.dim() > 2:
            logits_backbone = logits_backbone.view(-1, logits_backbone.size(-1))

        assert logits_agg.shape == logits_backbone.shape, \
            f"Shape mismatch: {logits_agg.shape} vs {logits_backbone.shape}"

        # Convert to probabilities
        log_probs_agg = F.log_softmax(logits_agg, dim=-1)
        probs_backbone = F.softmax(logits_backbone, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(log_probs_agg, probs_backbone, reduction='batchmean')

        return kl_loss

    def _compute_orthogonality_penalty(self, model) -> torch.Tensor:
        """
        Compute Frobenius orthogonality penalty from PEFT LoRA matrices.

        For now, we'll use a simple approach: penalize the norm of LoRA matrices
        to encourage sparsity. This is a placeholder for true orthogonality.

        Args:
            model: PEFT model with LoRA

        Returns:
            Orthogonality penalty tensor
        """
        penalty = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        # Collect all LoRA A matrices and compute pairwise orthogonality
        lora_A_matrices = []

        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'default'):
                # This is a LoRA layer
                lora_A = module.lora_A['default'].weight  # (rank, in_features)
                lora_A_matrices.append(lora_A)

        # Compute pairwise orthogonality penalty
        for i in range(len(lora_A_matrices)):
            for j in range(i + 1, len(lora_A_matrices)):
                A_i = lora_A_matrices[i]  # (rank, in_features)
                A_j = lora_A_matrices[j]  # (rank, in_features)

                # Only compare if they have the same shape
                if A_i.shape == A_j.shape:
                    # Compute ||A_i^T A_j||_F^2
                    cross_product = torch.mm(A_i, A_j.T)  # (rank, rank)
                    penalty = penalty + torch.norm(cross_product, p='fro') ** 2

        return penalty
