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
        bt_normalization_warmup: bool = False,
    ):
        super().__init__()
        self.P = P
        self.warmup_steps = warmup_steps
        self.design_layer = design_layer
        self.lambda_bt = lambda_bt
        self.bt_normalization_warmup = bt_normalization_warmup

        if self.bt_normalization_warmup:
            self.normalizer = RunningMeanNormalizer(target=20)  # NOTE: During warmup
        else:
            self.normalizer = None

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

        bt_loss = self._compute_standard_barlow_twins(hidden_states_by_stream)

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
            "loss/total": total_loss
        }

        return total_loss, loss_components

    def _compute_standard_barlow_twins(self, hidden_states_by_stream: torch.Tensor) -> torch.Tensor:
        """Frobenius Barlow Twins loss over all cross-stream pairs."""
        P, batch_size, seqlen, hidden_size = hidden_states_by_stream.shape
        device = hidden_states_by_stream.device
        assert P >= 2, f"Need at least 2 streams for Barlow Twins, got {P}"

        # Normalize representations
        reps_norm = []
        for p in range(P):
            rep = hidden_states_by_stream[p].view(-1, hidden_states_by_stream.size(-1))
            reps_norm.append((rep - rep.mean(0)) / (rep.std(0) + 1e-8))

        # Accumulate the off-diagonal cross-correlation loss for every stream pair
        sampled = []
        I = torch.eye(reps_norm[0].size(-1), device=device)
        for p in range(P):
            for q in range(p + 1, P):
                C = torch.mm(reps_norm[p].T, reps_norm[q]) / reps_norm[0].size(0)
                sampled.append(torch.norm(C - I, p='fro'))

        hidden_factor = hidden_size * (hidden_size - 1) / 2
        bt_loss = sum(sampled) / len(sampled) / hidden_factor * 4096

        return bt_loss
