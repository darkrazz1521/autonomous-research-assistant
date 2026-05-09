"""Lightweight BM25 scoring for scientific chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


class BM25Index:
    """Simple in-memory BM25 index."""

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents
        self.tokens = {doc_id: self._tokenize(text) for doc_id, text in documents.items()}
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.term_freqs: dict[str, Counter[str]] = {}
        self.avgdl = 0.0
        self._build()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9][a-z0-9_\-\.]+", text.lower())

    def _build(self) -> None:
        total_length = 0
        for doc_id, tokens in self.tokens.items():
            counts = Counter(tokens)
            self.term_freqs[doc_id] = counts
            total_length += len(tokens)
            for token in counts:
                self.doc_freqs[token] += 1
        self.avgdl = total_length / max(len(self.tokens), 1)

    def score(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores: dict[str, float] = defaultdict(float)
        doc_count = max(len(self.tokens), 1)
        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for doc_id, tf_counts in self.term_freqs.items():
                tf = tf_counts.get(token, 0)
                if tf == 0:
                    continue
                doc_len = max(len(self.tokens[doc_id]), 1)
                numerator = tf * 2.2
                denominator = tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / max(self.avgdl, 1.0))
                scores[doc_id] += idf * (numerator / denominator)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

