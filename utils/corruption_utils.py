"""
Text corruption utilities for contrastive learning.
Implements four corruption types: temporal inversion, numerical perturbation,
negation injection, and entity swaps using spaCy for NER.
"""

import logging
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import spacy
from datasets import load_dataset
from spacy.matcher import Matcher

logger = logging.getLogger(__name__)


class TextCorruptor:
    """
    Text corruption pipeline for generating negative examples in contrastive learning.
    Uses spaCy for NER and pattern matching.
    """

    def __init__(self, seed: int = 42, build_vocabulary: bool = True):
        """Initialize with spaCy model and corruption patterns."""
        self.rng = random.Random(seed)
        self.nlp = spacy.load("en_core_web_sm")

        # Setup pattern matchers
        self.temporal_matcher = Matcher(self.nlp.vocab)
        self._setup_temporal_patterns()

        # Corruption vocabulary - will be populated from dataset
        self.temporal_pairs = {}
        self.numbers = []
        self.entities_by_type = defaultdict(set)
        self.negation_targets = set()

        if build_vocabulary:
            self._build_corruption_vocabulary()

        # Corruption method mapping
        self.corruption_methods = {
            # "temporal": self._corrupt_temporal,
            "numerical": self._corrupt_numerical,
            "negation": self._corrupt_negation,
            "entity": self._corrupt_entity_swap
        }

        logger.info("TextCorruptor initialized with entities: numerical=%d, entities=%d",
                    len(self.numbers), sum(len(v) for v in self.entities_by_type.values()))

    def _setup_temporal_patterns(self):
        """Setup temporal pattern matching rules."""
        # Temporal relationship patterns
        temporal_patterns = [
            [{"LOWER": {"IN": ["after", "before", "since", "until", "during", "by"]}},
             {"IS_ALPHA": True, "OP": "*"}],
            [{"LOWER": {"IN": ["followed", "preceded"]}}, {"LOWER": "by"}],
            [{"SHAPE": "dddd"}, {"TEXT": "-"}, {"SHAPE": "dddd"}],  # Year ranges
            [{"LOWER": {"IN": ["then", "next", "previously", "later", "earlier"]}}]
        ]

        for i, pattern in enumerate(temporal_patterns):
            self.temporal_matcher.add(f"TEMPORAL_{i}", [pattern])

    def _build_corruption_vocabulary(self):
        """Build corruption vocabulary from The Pile dataset."""
        logger.info("Building corruption vocabulary from dataset...")

        try:
            # Load a small subset of The Pile for vocabulary extraction
            dataset = load_dataset("monology/pile-uncopyrighted",
                                   split="train", streaming=True,
                                   data_files=["train/00.jsonl.zst"])  # Just first shard

            max_docs = 1500  # Process 5K docs to build vocabulary
            for processed_docs, example in enumerate(dataset):
                if processed_docs >= max_docs:
                    break

                doc = self.nlp(example["text"])
                # self._extract_temporal_vocabulary(doc, text)
                self._extract_numerical_vocabulary(doc)
                self._extract_entity_vocabulary(doc)
                self._extract_negation_vocabulary(doc)

                processed_docs += 1
                if processed_docs % 500 == 0:
                    logger.info("Processed %d documents...", processed_docs)

            logger.info("Vocabulary building complete from %d documents", processed_docs)

        except Exception as e:
            logger.warning("Failed to build vocabulary from dataset: %s", e, exc_info=True)

    def _extract_temporal_vocabulary(self, doc, text: str):
        """Extract temporal words and phrases from document."""
        # Extract temporal expressions
        temporal_words = set()

        for token in doc:
            if token.lower_ in ["before", "after", "since", "until", "during", "then",
                                "next", "previous", "later", "earlier", "followed", "preceded"]:
                temporal_words.add(token.lower_)

        # Build temporal pairs dynamically
        temporal_opposites = {
            "before": "after", "after": "before",
            "since": "until", "until": "since",
            "earlier": "later", "later": "earlier",
            "previous": "next", "next": "previous",
            "followed": "preceded", "preceded": "followed"
        }

        for word in temporal_words:
            if word in temporal_opposites:
                self.temporal_pairs[word] = temporal_opposites[word]

    def _extract_numerical_vocabulary(self, doc):
        """Extract numbers from document."""
        for token in doc:
            if token.like_num and token.text.replace(",", "").replace(".", "").isdigit():
                # Parse and store number
                try:
                    num_text = token.text.replace(",", "")
                    if "." in num_text:
                        num_val = float(num_text)
                    else:
                        num_val = int(num_text)

                    if 0 < num_val < 1e10:  # Reasonable range
                        self.numbers.append(num_val)
                except ValueError as e:
                    logger.debug("Skipping invalid number token '%s': %s", token.text, e)
                    continue

        # Keep only unique numbers, limit size
        self.numbers = list(set(self.numbers))
        if len(self.numbers) > 2000:
            self.numbers = self.rng.sample(self.numbers, 2000)

    def _extract_entity_vocabulary(self, doc):
        """Extract named entities from document."""
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "DATE"] and len(ent.text.strip()) > 1:
                # Clean entity text
                entity_text = ent.text.strip()
                if entity_text.isalpha() or ent.label_ == "DATE":  # Allow letters or dates
                    self.entities_by_type[ent.label_].add(entity_text)

        # Limit entity vocabulary size per type
        for entity_type in self.entities_by_type:
            entities = list(self.entities_by_type[entity_type])
            if len(entities) > 1000:
                self.entities_by_type[entity_type] = set(self.rng.sample(entities, 1000))

    def _extract_negation_vocabulary(self, doc):
        """Extract verbs that could be targets for negation."""
        for token in doc:
            if (token.pos_ == "VERB" and
                token.lower_ not in ["is", "are", "was", "were", "be", "been", "being"] and
                    len(token.text) > 2):
                self.negation_targets.add(token.lower_)

        # Limit negation targets
        if len(self.negation_targets) > 1500:
            targets = list(self.negation_targets)
            self.negation_targets = set(self.rng.sample(targets, 1500))

    def corrupt_text(self, text: str, max_corruptions: int = 2) -> Tuple[str, List[str]]:
        """
        Apply 1-2 random corruptions to text.

        Args:
            text: Input text to corrupt
            max_corruptions: Maximum number of corruption types to apply

        Returns:
            Tuple of (corrupted_text, list_of_corruption_types_applied)
        """
        assert text is not None, "Text input cannot be None"
        if not text.strip():
            return text, []

        # Parse text with spaCy
        doc = self.nlp(text[:self.nlp.max_length])

        # Determine available corruption methods
        available_methods = []

        # Check for temporal patterns
        # if self.temporal_matcher(doc):
        #     available_methods.append("temporal")

        # Check for numbers
        if any(token.like_num for token in doc):
            available_methods.append("numerical")

        # Check for negation opportunities (always available)
        available_methods.append("negation")

        # Check for named entities
        entities = [ent for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "ORG", "DATE"]]
        if entities:
            available_methods.append("entity")

        if not available_methods:
            logger.warning("No corruption methods available for text")
            return text, []

        # Select 1-2 corruption methods
        num_corruptions = self.rng.randint(1, min(max_corruptions, len(available_methods)))
        selected_methods = self.rng.sample(available_methods, num_corruptions)

        # Apply corruptions sequentially
        corrupted_text = text
        applied_corruptions = []

        for method in selected_methods:
            try:
                new_text, success = self.corruption_methods[method](corrupted_text)
                if success:
                    corrupted_text = new_text
                    applied_corruptions.append(method)
            except Exception as e:
                logger.warning("Failed to apply %s corruption: %s", method, e, exc_info=True)

        return corrupted_text, applied_corruptions

    def _corrupt_temporal(self, text: str) -> Tuple[str, bool]:
        """Apply temporal inversion corruption using dynamic vocabulary."""
        corrupted = text
        success = False

        # Use dynamic temporal pairs from vocabulary
        if self.temporal_pairs:
            available_swaps = [(orig, repl) for orig, repl in self.temporal_pairs.items()
                               if orig in text.lower()]

            if available_swaps:
                original, replacement = self.rng.choice(available_swaps)
                # Case-preserving replacement
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                corrupted = pattern.sub(replacement, corrupted, count=1)
                success = True

        # Handle year ranges (e.g., "1914-1918" -> "1918-1914")
        if not success:
            year_pattern = r'\b(\d{4})-(\d{4})\b'
            match = re.search(year_pattern, text)
            if match:
                year1, year2 = match.groups()
                corrupted = re.sub(year_pattern, "%s-%s" % (year2, year1), text, count=1)
                success = True

        return corrupted, success

    def _corrupt_numerical(self, text: str) -> Tuple[str, bool]:
        """Apply numerical replacement corruption using dynamic vocabulary."""
        # Find numbers in text
        number_pattern = r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b'
        matches = list(re.finditer(number_pattern, text))

        assert self.numbers is not None, "Numbers vocabulary not initialized"
        if not matches or not self.numbers:
            return text, False

        # Select random number to modify
        match = self.rng.choice(matches)
        original_str = match.group(1)

        # Parse original number (handle commas)
        try:
            original_num = float(original_str.replace(',', ''))
        except ValueError as e:
            logger.warning("Failed to parse number '%s': %s", original_str, e)
            return text, False

        # Filter numbers from vocabulary that are different from original
        available_numbers = [n for n in self.numbers if abs(n - original_num) > 0.01]

        if not available_numbers:
            logger.debug("No suitable replacement numbers found for %f", original_num)
            return text, False

        # Choose replacement from dynamic vocabulary
        replacement_num = self.rng.choice(available_numbers)

        # Format replacement number to match original format
        if '.' in original_str:
            # Preserve decimal places for floating point
            decimal_places = len(original_str.split('.')[1])
            new_str = "{:,.{}f}".format(replacement_num, decimal_places)
        else:
            # Integer formatting
            new_str = "{:,}".format(int(replacement_num))

        # Replace in text
        corrupted = text[:match.start(1)] + new_str + text[match.end(1):]
        return corrupted, True

    def _corrupt_negation(self, text: str) -> Tuple[str, bool]:
        """Apply negation injection/removal corruption."""
        doc = self.nlp(text)

        # Look for verbs to negate or existing negations to remove
        for token in doc:
            # Remove existing negations
            if token.lower_ in ["not", "n't", "never", "no"] and token.i > 0:
                # Remove negation
                start_char = token.idx
                end_char = start_char + len(token.text)

                # Handle contractions like "don't" -> "do"
                if token.lower_ == "n't":
                    # Find the word this contraction is attached to
                    prev_token = doc[token.i - 1]
                    if prev_token.lower_ in ["do", "does", "did", "can", "could", "will", "would", "should"]:
                        start_char = prev_token.idx
                        replacement = prev_token.text.rstrip("n't")
                    else:
                        continue
                else:
                    replacement = ""

                corrupted = text[:start_char] + replacement + text[end_char:].lstrip()
                return corrupted.strip(), True

            # Add negation to verbs using dynamic vocabulary
            elif (token.pos_ == "VERB" and
                  token.lower_ in self.negation_targets and
                  token.lower_ not in ["is", "are", "was", "were"]):
                # Insert "not" after verb
                insert_pos = token.idx + len(token.text)
                corrupted = text[:insert_pos] + " not" + text[insert_pos:]
                return corrupted, True

        # Fallback: add "not" before first verb from dynamic vocabulary
        for token in doc:
            if token.pos_ == "VERB" and token.lower_ in self.negation_targets:
                insert_pos = token.idx
                corrupted = text[:insert_pos] + "not " + text[insert_pos:]
                return corrupted, True

        return text, False

    def _corrupt_entity_swap(self, text: str) -> Tuple[str, bool]:
        """Apply entity swap corruption using dynamic vocabulary."""
        doc = self.nlp(text)

        # Find swappable entities
        for ent in doc.ents:
            assert self.entities_by_type is not None, "Entity vocabulary not initialized"
            if ent.label_ in self.entities_by_type and len(self.entities_by_type[ent.label_]) > 0:
                # Choose replacement entity of same type from dynamic vocabulary
                available_entities = list(self.entities_by_type[ent.label_])

                # Filter out the current entity text to avoid no-op replacement
                available_entities = [e for e in available_entities if e.lower() != ent.text.lower()]

                if available_entities:
                    replacement = self.rng.choice(available_entities)
                    assert replacement != ent.text, "Replacement entity identical to original"

                    # Replace entity
                    start_char = ent.start_char
                    end_char = ent.end_char
                    corrupted = text[:start_char] + replacement + text[end_char:]
                    return corrupted, True

        return text, False
