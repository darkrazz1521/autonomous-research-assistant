"""Helpers for paper path resolution, token estimation, and text analytics."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are", "by", "we", "this",
    "that", "our", "be", "as", "from", "at", "can", "it", "using", "used", "into", "than", "such",
}


def safe_paper_id(arxiv_id: str) -> str:
    """Return a filesystem-safe paper id."""
    return arxiv_id.replace("/", "_")


def estimate_token_count(text: str) -> int:
    """Estimate token count without introducing an embedding tokenizer dependency."""
    words = len(re.findall(r"\S+", text))
    return max(1, int(words * 1.25))


def stable_text_hash(text: str) -> str:
    """Return a stable hash for deduplication."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def semantic_density_score(text: str) -> float:
    """Estimate semantic density from lexical variety and content-bearing tokens."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text)
    if not tokens:
        return 0.0
    content_tokens = [token.lower() for token in tokens if token.lower() not in STOPWORDS]
    unique_ratio = len(set(content_tokens)) / max(len(content_tokens), 1)
    long_token_ratio = sum(1 for token in content_tokens if len(token) >= 6) / max(len(content_tokens), 1)
    digit_signal = min(1.0, len(re.findall(r"\d", text)) / 20.0)
    return round(min(1.0, (unique_ratio * 0.5) + (long_token_ratio * 0.35) + (digit_signal * 0.15)), 4)


def topic_signature(text: str, limit: int = 6) -> list[str]:
    """Extract a lightweight topic signature for future retrieval metadata."""
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)]
    filtered = [token for token in tokens if token not in STOPWORDS]
    counts = Counter(filtered)
    return [token for token, _ in counts.most_common(limit)]
