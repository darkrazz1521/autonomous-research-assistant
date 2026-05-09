"""Noise scoring heuristics for retrieval chunks."""

from __future__ import annotations

import re
from collections import Counter


def numeric_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(char.isdigit() for char in text) / len(text), 6)


def alphabetic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(char.isalpha() for char in text) / len(text), 6)


def duplicate_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicated = sum(count - 1 for count in counts.values() if count > 1)
    return round(duplicated / len(lines), 6)


def citation_density(text: str) -> float:
    citations = re.findall(r"\[[^\]]+\]|\([A-Z][A-Za-z]+ et al\., \d{4}[a-z]?\)", text)
    tokens = max(len(re.findall(r"\w+", text)), 1)
    return round(len(citations) / tokens, 6)


def equation_density(text: str) -> float:
    equations = re.findall(r"[=<>±∑∏λμσ]|\\begin\{equation\}|\\frac|\bO\([^)]+\)", text)
    tokens = max(len(re.findall(r"\w+", text)), 1)
    return round(len(equations) / tokens, 6)


def classify_noise(*, semantic_density: float, table_probability: float, benchmark_probability: float, duplicate_ratio: float, numeric_share: float, malformed_structure: float) -> tuple[float, list[str]]:
    labels: list[str] = []
    if table_probability >= 0.55:
        labels.append("table_like")
    if benchmark_probability >= 0.55:
        labels.append("benchmark_heavy")
    if semantic_density < 0.35:
        labels.append("low_semantic_density")
    if duplicate_ratio > 0.20:
        labels.append("duplicate_lines")
    if numeric_share > 0.18:
        labels.append("numeric_wall")
    if malformed_structure > 0.35:
        labels.append("malformed_structure")
    score = (
        (1 - semantic_density) * 0.25
        + table_probability * 0.20
        + benchmark_probability * 0.15
        + duplicate_ratio * 0.15
        + min(numeric_share / 0.35, 1.0) * 0.10
        + malformed_structure * 0.15
    )
    return round(max(0.0, min(1.0, score)), 6), labels
