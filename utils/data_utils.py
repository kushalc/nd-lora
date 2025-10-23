"""
Data utilities for ParScale experiments using The Pile dataset.
Uses monology/pile-uncopyrighted from HuggingFace for streaming data loading.
Handles streaming, tokenization, and reproducible data loading.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

from utils.corruption_utils import TextCorruptor

logger = logging.getLogger(__name__)


def setup_pile_streaming(
    tokenizer: PreTrainedTokenizer,
    shard_ids: List[int],
    seq_len: int = 1024,
    batch_size: int = 1,
    num_workers: int = 1,
    skip_examples: int = 0,
    corruption=False,
    seed=42,
) -> Tuple[Any, Any]:
    # Use optimized streaming with batch tokenization
    dataset = create_pile_dataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        shard_ids=shard_ids,
        skip_examples=skip_examples,
        corruption=corruption,
        seed=seed,
    )
    loader = create_dataloader(dataset,
                               batch_size=batch_size,
                               num_workers=min(num_workers, len(shard_ids)))
    return loader


def create_pile_dataset(
    tokenizer: PreTrainedTokenizer,
    shard_ids: List[int],
    seq_len: int = 1024,
    seed: int = 42,
    skip_examples: int = 0,
    corruption=False,
):
    """
    Create a simple streaming dataset from The Pile.

    Args:
        tokenizer: Tokenizer to use
        seq_len: Maximum sequence length  
        seed: Random seed for shuffling
        skip_examples: Number of examples to skip (for validation split)

    Returns:
        Generator that yields tokenized examples
    """
    # Load the dataset
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
        data_files=["train/%02d.jsonl.zst" % idx for idx in shard_ids],
    )

    # Shuffle with seed
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    # Skip examples if needed (for validation)
    if skip_examples > 0:
        dataset = dataset.skip(skip_examples)

    corruptor = None
    if corruption:
        corruptor = TextCorruptor(seed=seed)

    # Process and yield examples
    for example in dataset:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt"
        )

        yield {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze(),  # For causal LM
            "is_corrupted": False,
            # "corruption_types": [],
            # "original_text": example["text"],
            # "corrupted_text": None
        }

        if corruption:
            corrupted_text, corruption_types = corruptor.corrupt_text(example["text"])
            tokenized = tokenizer(
                corrupted_text,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt"
            )
            yield {
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": tokenized["input_ids"].squeeze(),
                "is_corrupted": True,
                # "corruption_types": corruption_types,
                # "original_text": example["text"],
                # "corrupted_text": corrupted_text
            }


def create_dataloader(
    dataset,  # Generator from create_pile_dataset
    batch_size: int = 1,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for streaming dataset.

    Args:
        dataset: Generator from create_pile_dataset
        batch_size: Batch size (typically 1 for gradient accumulation)
        num_workers: Number of worker processes (must be 0 for generators)

    Returns:
        DataLoader instance
    """
    # Convert generator to IterableDataset
    class SimpleIterableDataset(IterableDataset):
        def __init__(self, generator_func):
            self.generator_func = generator_func

        def __iter__(self):
            return self.generator_func

    iterable_dataset = SimpleIterableDataset(dataset)

    return DataLoader(
        iterable_dataset,
        batch_size=batch_size,
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=torch.cuda.is_available() and num_workers > 0,  # Only pin_memory with workers
        persistent_workers=True if num_workers > 0 else False
    )


def count_tokens_processed(
    batch_size: int,
    target_tokens: int,
    seq_len: int,
    grad_accumulation_steps: int = 1
) -> Tuple[int, int]:
    """
    Calculate number of training steps needed to reach target token budget.

    Args:
        dataloader: DataLoader instance (only used for batch_size)
        target_tokens: Target number of tokens to process
        seq_len: Sequence length per example
        grad_accumulation_steps: Number of gradient accumulation steps

    Returns:
        Tuple of (total_steps, actual_tokens)
    """
    tokens_per_batch = seq_len * batch_size

    # Note: Calculating scheduler steps based on token accumulation
    tokens_per_step = tokens_per_batch * grad_accumulation_steps
    total_steps = target_tokens // tokens_per_step + 1
    return total_steps


def select_pile_shards(num_train_shards: int = 1, total_available: int = 30) -> List[int]:
    """
    Select random Pile shards for reproducible experiments.

    Args:
        num_shards: Number of shards to select
        seed: Random seed for selection
        total_available: Total number of available shards

    Returns:
        List of selected shard IDs
    """
    available_shards = list(range(total_available))
    train_shard_ids = random.sample(available_shards, min(num_train_shards, len(available_shards)))

    available_shards = list(set(range(total_available)) - set(train_shard_ids))
    num_val_shards = min(int(max(1, np.ceil(num_train_shards / 10))), len(available_shards))
    val_shard_ids = random.sample(available_shards, min(num_val_shards, total_available))
    logging.info("Selected train_shard_ids=%s val_shard_ids=%s", train_shard_ids, val_shard_ids)
    return train_shard_ids, val_shard_ids


def get_tokenizer_info(tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Get tokenizer information for logging.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Dictionary of tokenizer metadata
    """
    special_tokens = {}
    if hasattr(tokenizer, 'special_tokens_map'):
        special_tokens = tokenizer.special_tokens_map

    return {
        "name": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": getattr(tokenizer, 'model_max_length', None),
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
        "special_tokens": special_tokens
    }


class TokenBudgetTracker:
    """
    Track token consumption during training to ensure equal budgets across P values.
    """

    def __init__(self, target_tokens: int, seq_len: int, batch_size: int):
        self.target_tokens = target_tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.processed_tokens = 0
        self.processed_steps = 0

    def update(self, batch_size: Optional[int] = None) -> bool:
        """
        Update token count and check if target is reached.

        Args:
            batch_size: Override batch size for this step

        Returns:
            True if target tokens reached, False otherwise
        """
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        step_tokens = self.seq_len * effective_batch_size

        self.processed_tokens += step_tokens
        self.processed_steps += 1

        return self.processed_tokens >= self.target_tokens

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        return {
            "processed_tokens": self.processed_tokens,
            "target_tokens": self.target_tokens,
            "progress_pct": self.processed_tokens / self.target_tokens,
            "processed_steps": self.processed_steps,
            "remaining_tokens": max(0, self.target_tokens - self.processed_tokens)
        }

    def is_complete(self) -> bool:
        """Check if target tokens have been reached."""
        return self.processed_tokens >= self.target_tokens


def collate_contrastive_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for contrastive batches.
    Groups corrupted and original examples separately for loss computation.

    Args:
        batch: List of dataset examples

    Returns:
        Collated batch with separated corrupted/original examples
    """
    import torch

    # Separate corrupted and original examples
    corrupted_examples = [ex for ex in batch if ex["is_corrupted"]]
    original_examples = [ex for ex in batch if not ex["is_corrupted"]]

    def stack_examples(examples: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        if not examples:
            return None

        return {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
            "labels": torch.stack([ex["labels"] for ex in examples])
        }

    # Stack examples by type
    corrupted_batch = stack_examples(corrupted_examples)
    original_batch = stack_examples(original_examples)

    # Collect corruption metadata
    corruption_metadata = {
        "corruption_types": [ex["corruption_types"] for ex in corrupted_examples],
        "num_corrupted": len(corrupted_examples),
        "num_original": len(original_examples)
    }

    return {
        "corrupted": corrupted_batch,
        "original": original_batch,
        "metadata": corruption_metadata,
        "batch_size": len(batch)
    }


def get_contrastive_stats(dataloader, num_batches: int = 100) -> Dict[str, Any]:
    """
    Analyze contrastive dataset statistics.

    Args:
        dataloader: Contrastive DataLoader
        num_batches: Number of batches to analyze

    Returns:
        Statistics dictionary
    """
    corruption_counts = {"temporal": 0, "numerical": 0, "negation": 0, "entity": 0}
    total_corrupted = 0
    total_original = 0

    logger.info(f"Analyzing {num_batches} batches for contrastive statistics")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        for example in batch:
            if example["is_corrupted"]:
                total_corrupted += 1
                for corruption_type in example["corruption_types"]:
                    corruption_counts[corruption_type] += 1
            else:
                total_original += 1

    total_examples = total_corrupted + total_original

    stats = {
        "total_examples": total_examples,
        "corrupted_examples": total_corrupted,
        "original_examples": total_original,
        "corruption_ratio": total_corrupted / total_examples if total_examples > 0 else 0.0,
        "corruption_type_counts": corruption_counts,
        "avg_corruptions_per_example": sum(corruption_counts.values()) / max(total_corrupted, 1)
    }

    logger.info(f"Dataset stats: {stats}")
    return stats
