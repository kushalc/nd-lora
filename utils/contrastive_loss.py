"""
Contrastive loss functions for ParScale stream variance training.
Implements variance-based disagreement loss to encourage stream consensus on factual
content and disagreement on corrupted content.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_stream_variance(stream_logits: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute variance across ParScale streams using memory-efficient Welford's algorithm.

    Args:
        stream_logits: List of logits tensors from P streams
                      Each tensor: [batch_size, seq_len, vocab_size]

    Returns:
        Stream disagreement scores: [batch_size, seq_len]
        Higher values indicate more disagreement between streams
    """
    if len(stream_logits) < 2:
        raise ValueError(f"Need at least 2 streams for variance, got {len(stream_logits)}")

    # Welford's online algorithm for memory-efficient variance computation
    # Initialize with first stream
    mean = stream_logits[0].clone()
    m2 = torch.zeros_like(mean)

    # Incrementally update mean and variance
    for i, logits in enumerate(stream_logits[1:], start=2):
        delta = logits - mean
        mean += delta / i
        delta2 = logits - mean
        m2 += delta * delta2

    # Final variance (divide by n for population variance)
    variance = m2 / len(stream_logits)

    # Sum over vocabulary dimension to get disagreement scores
    disagreement_scores = torch.sum(variance, dim=-1)

    return disagreement_scores


def compute_stream_entropy_variance(stream_logits: List[torch.Tensor]) -> torch.Tensor:
    """
    Alternative: Compute variance of stream entropies instead of logit variance.

    Args:
        stream_logits: List of logits tensors from P streams

    Returns:
        Entropy disagreement scores: [batch_size, seq_len]
    """
    # Compute entropy for each stream
    stream_entropies = []
    for logits in stream_logits:
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        stream_entropies.append(entropy)

    # Stack and compute variance across streams
    stacked_entropies = torch.stack(stream_entropies, dim=0)  # [P, batch_size, seq_len]
    entropy_variance = torch.var(stacked_entropies, dim=0, unbiased=False)

    return entropy_variance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training ParScale streams to exhibit intelligent disagreement.
    Combines cross-entropy loss with variance-based contrastive term.
    """

    def __init__(
        self,
        gamma: float = 0.5,
        variance_mode: str = "logits",
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        """
        Initialize contrastive loss.

        Args:
            gamma: Weight for contrastive term (0.0 disables contrastive learning)
            variance_mode: "logits" or "entropy" for variance computation
            reduction: "mean" or "sum" for loss reduction
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.gamma = gamma
        self.variance_mode = variance_mode
        self.reduction = reduction

        # Standard cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing
        )

        logger.info(f"ContrastiveLoss initialized: gamma={gamma}, variance_mode={variance_mode}")

    def forward(
        self,
        stream_logits: List[torch.Tensor],
        labels: torch.Tensor,
        is_corrupted: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive loss.

        Args:
            stream_logits: List of logits from P streams [batch, seq_len, vocab]
            labels: Target token IDs [batch, seq_len] 
            is_corrupted: Boolean mask indicating corrupted examples [batch]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if len(stream_logits) == 0:
            raise ValueError("Empty stream_logits list")

        batch_size, seq_len, vocab_size = stream_logits[0].shape

        # Compute standard cross-entropy loss using first stream
        # (all streams should predict the same for factual content)
        ce_loss = self.ce_loss(
            stream_logits[0].view(-1, vocab_size),
            labels.view(-1)
        )

        # Initialize loss components
        loss_components = {
            "ce_loss": ce_loss,
            "contrastive_loss": torch.tensor(0.0, device=ce_loss.device),
            "disagreement_factual": torch.tensor(0.0, device=ce_loss.device),
            "disagreement_corrupted": torch.tensor(0.0, device=ce_loss.device)
        }

        # Skip contrastive loss if gamma=0 or single stream
        if self.gamma == 0.0 or len(stream_logits) < 2:
            total_loss = ce_loss
        else:
            # Compute stream disagreement
            if self.variance_mode == "logits":
                disagreement = compute_stream_variance(stream_logits)
            elif self.variance_mode == "entropy":
                disagreement = compute_stream_entropy_variance(stream_logits)
            else:
                raise ValueError(f"Unknown variance_mode: {self.variance_mode}")

            # Apply attention mask if provided
            if attention_mask is not None:
                disagreement = disagreement * attention_mask
                valid_tokens = attention_mask.sum()
            else:
                valid_tokens = batch_size * seq_len

            # Compute contrastive loss based on corruption labels
            if is_corrupted is not None:
                # Separate factual vs corrupted examples
                factual_mask = ~is_corrupted  # [batch]
                corrupted_mask = is_corrupted  # [batch]

                if factual_mask.any():
                    # Minimize disagreement on factual content
                    factual_disagreement = disagreement[factual_mask].mean()
                    loss_components["disagreement_factual"] = factual_disagreement
                else:
                    factual_disagreement = torch.tensor(0.0, device=disagreement.device)

                if corrupted_mask.any():
                    # Maximize disagreement on corrupted content (minimize negative)
                    corrupted_disagreement = disagreement[corrupted_mask].mean()
                    loss_components["disagreement_corrupted"] = corrupted_disagreement
                else:
                    corrupted_disagreement = torch.tensor(0.0, device=disagreement.device)

                # Contrastive loss: minimize factual disagreement, maximize corrupted disagreement
                # NOTE: Normalize for better behavior with standard cross-entropy loss.
                contrastive_loss = 10 * (factual_disagreement - corrupted_disagreement) / factual_disagreement
            else:
                logging.warning("Couldn't find contrast; defaulting to zero contrastive loss")
                contrastive_loss = 0

            loss_components["contrastive_loss"] = contrastive_loss
            total_loss = ce_loss + self.gamma * contrastive_loss

        return total_loss, loss_components

    def compute_calibration_metrics(
        self,
        stream_logits: List[torch.Tensor],
        labels: torch.Tensor,
        is_corrupted: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute calibration metrics for stream behavior analysis.

        Args:
            stream_logits: Stream logits
            labels: True labels
            is_corrupted: Corruption indicators

        Returns:
            Dictionary of calibration metrics
        """
        if len(stream_logits) < 2:
            return {}

        # Compute disagreement scores
        if self.variance_mode == "logits":
            disagreement = compute_stream_variance(stream_logits)
        else:
            disagreement = compute_stream_entropy_variance(stream_logits)

        # Compute accuracy for each position
        predictions = torch.argmax(stream_logits[0], dim=-1)
        accuracy = (predictions == labels).float()

        # Average over sequence length
        seq_disagreement = disagreement.mean(dim=-1)  # [batch]
        seq_accuracy = accuracy.mean(dim=-1)  # [batch]

        # Separate by corruption status
        factual_mask = ~is_corrupted
        corrupted_mask = is_corrupted

        metrics = {}

        if factual_mask.any():
            metrics.update({
                "factual_disagreement_mean": seq_disagreement[factual_mask].mean(),
                "factual_disagreement_std": seq_disagreement[factual_mask].std(),
                "factual_accuracy_mean": seq_accuracy[factual_mask].mean()
            })

        if corrupted_mask.any():
            metrics.update({
                "corrupted_disagreement_mean": seq_disagreement[corrupted_mask].mean(),
                "corrupted_disagreement_std": seq_disagreement[corrupted_mask].std(),
                "corrupted_accuracy_mean": seq_accuracy[corrupted_mask].mean()
            })

        # Overall metrics
        metrics.update({
            "overall_disagreement_mean": seq_disagreement.mean(),
            "overall_accuracy_mean": seq_accuracy.mean(),
            "disagreement_accuracy_corr": torch.corrcoef(
                torch.stack([seq_disagreement, seq_accuracy])
            )[0, 1] if len(seq_disagreement) > 1 else torch.tensor(0.0)
        })

        return metrics


def extract_parscale_stream_logits(model_output, P: int) -> List[torch.Tensor]:
    """
    Extract individual stream logits from ParScale model output.

    Args:
        model_output: Output from ParScale model
        P: Number of parallel streams

    Returns:
        List of stream logits tensors, each of shape [batch_size, seq_len, vocab_size]
    """
    if hasattr(model_output, 'stream_logits') and model_output.stream_logits is not None:
        # Model directly provides individual stream logits
        return model_output.stream_logits
    else:
        # Fallback for P=1 or models without stream output capability
        # Duplicate main logits for each "stream"
        num_streams = max(1, P)
        logger.warning(f"No individual stream logits available - duplicating main logits {num_streams} times")
        return [model_output.logits] * num_streams
