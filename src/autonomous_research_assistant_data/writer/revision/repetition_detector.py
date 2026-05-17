"""Detect repetition across sections and within a draft."""

from __future__ import annotations

import re


class RepetitionDetector:
    """Score repeated phrases, sentences, and topic explanations."""

    def _sentences(self, text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    def repeated_sentences(self, text: str) -> list[str]:
        seen: set[str] = set()
        repeated: list[str] = []
        for sentence in self._sentences(text):
            key = re.sub(r"\s+", " ", sentence.lower())
            if key in seen and key not in repeated:
                repeated.append(key)
            seen.add(key)
        return repeated

    def repeated_ngrams(self, text: str, n: int = 4) -> list[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())
        counts: dict[str, int] = {}
        for index in range(0, max(len(tokens) - n + 1, 0)):
            gram = " ".join(tokens[index : index + n])
            counts[gram] = counts.get(gram, 0) + 1
        return [gram for gram, count in counts.items() if count > 1]

    def cross_section_overlap(self, text: str, previous_texts: list[str]) -> float:
        if not text or not previous_texts:
            return 0.0
        current = set(re.findall(r"[a-z][a-z0-9\-]{3,}", text.lower()))
        if not current:
            return 0.0
        prior = set()
        for item in previous_texts:
            prior.update(re.findall(r"[a-z][a-z0-9\-]{3,}", item.lower()))
        return round(len(current.intersection(prior)) / max(len(current), 1), 6)

    def discourse_diversity(self, text: str) -> float:
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / max(len(tokens), 1), 6)

