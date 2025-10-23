"""
Stream-aware LoRA implementation for ParScale.
Each stream gets independent LoRA parameters while maintaining full attention across all samples.
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
from peft.tuners import lora

logger = logging.getLogger(__name__)


def parse_streams_from_batch(hidden_states: torch.Tensor, P: int) -> torch.Tensor:
    """
    Parse batched hidden states into per-stream tensors.

    Args:
        hidden_states: Batched tensor (P*batch_per_stream, ...)
        P: Number of parallel streams

    Returns:
        Per-stream tensor (P, batch_per_stream, ...)
    """
    assert hidden_states.shape[0] % P == 0, f"Batch size {hidden_states.shape[0]} not divisible by P={P}"
    batch_per_stream = hidden_states.shape[0] // P

    # Reshape (P*batch_per_stream, ...) -> (P, batch_per_stream, ...)
    new_shape = (P, batch_per_stream) + hidden_states.shape[1:]
    return hidden_states.view(new_shape)


class StreamAwareLoRA(nn.Module):
    """
    LoRA adapter that applies different weights to different streams.
    All samples attend to each other, but use different LoRA parameters.
    """

    def __init__(
        self,
        original_layer: lora.Linear,
        P: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        device: str = None,
    ):
        super().__init__()
        self.base_layer = original_layer.base_layer

        assert rank % P == 0
        self.P = P
        self.rank_per_stream = rank
        self.alpha = alpha
        self.scaling = alpha / self.rank_per_stream

        self.in_features = in_features = self.base_layer.in_features
        self.out_features = out_features = self.base_layer.out_features

        # Get device from the original layer's weights if not specified
        if device is None:
            device = self.base_layer.weight.device

        # Create P independent LoRA weight sets
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(self.rank_per_stream, in_features, device=device))
            for _ in range(P)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, self.rank_per_stream, device=device))
            for _ in range(P)
        ])

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        # Use small random Gaussian for B to ensure non-zero variance from start
        for p in range(P):
            nn.init.kaiming_uniform_(self.lora_A[p], a=math.sqrt(5))
            nn.init.normal_(self.lora_B[p], mean=0.0, std=1e-4)

        # Freeze base layer
        for param in original_layer.parameters():
            param.requires_grad = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying stream-specific LoRA.

        Args:
            hidden_states: (P*batch_size, seq_len, hidden_dim) or (P*batch_size, hidden_dim)
        """
        base_states = self.base_layer(hidden_states)
        hidden_states_dropout = self.lora_dropout(hidden_states)

        # Parse streams from batched input
        hidden_states_by_stream = parse_streams_from_batch(hidden_states_dropout, self.P)

        lora_outputs = []
        for p in range(self.P):
            stream_hidden = hidden_states_by_stream[p]

            # Apply this stream's LoRA: (BA)x
            lora_a_out = torch.matmul(stream_hidden, self.lora_A[p].T)
            lora_b_out = torch.matmul(lora_a_out, self.lora_B[p].T)
            lora_outputs.append(lora_b_out * self.scaling)

        # Concatenate all stream outputs back to original batch order
        lora_output = torch.cat(lora_outputs, dim=0)

        assert lora_output.shape == base_states.shape
        return base_states + lora_output


def replace_linear_with_stream_lora(
    model: nn.Module,
    P: int,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1
) -> int:
    """
    Replace specified Linear layers with StreamAwareLoRA.

    Returns:
        Number of modules replaced
    """
    modules_replaced = 0
    replacements = []

    # Collect modules to replace (can't modify dict during iteration)
    for name, module in model.named_modules():
        if not isinstance(module, lora.Linear):
            continue
        replacements.append((name, module))

    # Perform replacements
    for name, module in replacements:
        # Get parent module and attribute name
        *parent_parts, attr_name = name.split('.')
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)

        # Create StreamAwareLoRA replacement
        stream_lora = StreamAwareLoRA(original_layer=module, P=P, rank=rank,
                                      alpha=alpha, dropout=dropout)

        # Replace the module
        setattr(parent, attr_name, stream_lora)
        modules_replaced += 1

        logger.debug("Replaced %s with StreamAwareLoRA", name)

    assert modules_replaced > 0
    return model
