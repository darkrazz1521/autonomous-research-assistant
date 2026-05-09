"""Semantic-density heuristics for scientific chunks."""

from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\./%]*")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "these",
    "those",
}


def tokenise(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def semantic_density_score(text: str) -> float:
    tokens = tokenise(text)
    if not tokens:
        return 0.0
    lexical = [token.lower() for token in tokens if any(char.isalpha() for char in token)]
    if not lexical:
        return 0.0
    content_ratio = sum(token not in STOPWORDS for token in lexical) / len(lexical)
    unique_ratio = len(set(lexical)) / len(lexical)
    long_token_ratio = sum(len(token) >= 6 for token in lexical) / len(lexical)
    sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
    sentence_balance = min(len(lexical) / (sentence_count * 18), 1.0)
    score = (content_ratio * 0.35) + (unique_ratio * 0.25) + (long_token_ratio * 0.20) + (sentence_balance * 0.20)
    return round(max(0.0, min(1.0, score)), 6)


def language_entropy(text: str) -> float:
    letters = [char.lower() for char in text if char.isalpha()]
    if not letters:
        return 0.0
    counts = Counter(letters)
    total = len(letters)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    normalized = entropy / math.log2(min(26, len(counts)) or 1)
    return round(max(0.0, min(1.0, normalized)), 6)


def average_sentence_length(text: str) -> float:
    sentences = [segment.strip() for segment in re.split(r"[.!?]+\s+", text) if segment.strip()]
    if not sentences:
        return 0.0
    lengths = [len(tokenise(sentence)) for sentence in sentences]
    return round(sum(lengths) / max(len(lengths), 1), 6)
