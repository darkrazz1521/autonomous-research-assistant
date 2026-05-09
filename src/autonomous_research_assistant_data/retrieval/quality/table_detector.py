"""Heuristics for detecting table-like and benchmark-like chunk structure."""

from __future__ import annotations

import re


ROW_PATTERN = re.compile(r"^\s*(?:[\w\-\+/\.]+\s+){2,}[\d\.%]+(?:\s+[\d\.%]+){2,}\s*$")
BENCHMARK_TERMS = {
    "accuracy",
    "f1",
    "score",
    "scores",
    "benchmark",
    "leaderboard",
    "winrate",
    "success",
    "pass@1",
    "metric",
    "results",
}


def detect_table_probability(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    row_hits = sum(bool(ROW_PATTERN.match(line)) for line in lines)
    delimiter_hits = sum(line.count("|") >= 2 or line.count("\t") >= 2 for line in lines)
    dense_numeric_lines = sum(sum(char.isdigit() for char in line) / max(len(line), 1) > 0.22 for line in lines)
    probability = ((row_hits / len(lines)) * 0.5) + ((delimiter_hits / len(lines)) * 0.25) + ((dense_numeric_lines / len(lines)) * 0.25)
    return round(max(0.0, min(1.0, probability)), 6)


def detect_benchmark_probability(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    term_hits = sum(any(term in line for term in BENCHMARK_TERMS) for line in lines)
    percentage_hits = sum(("%" in line) or bool(re.search(r"\b\d+\.\d+\b", line)) for line in lines)
    probability = ((term_hits / len(lines)) * 0.55) + ((percentage_hits / len(lines)) * 0.45)
    return round(max(0.0, min(1.0, probability)), 6)


def malformed_structure_score(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    fragmented = sum(len(line.split()) <= 3 for line in lines) / len(lines)
    broken_case = sum(bool(re.match(r"^[a-z].*[A-Z]", line)) for line in lines) / len(lines)
    punctuation_noise = sum(line.count("(") != line.count(")") or line.count("[") != line.count("]") for line in lines) / len(lines)
    score = (fragmented * 0.45) + (broken_case * 0.25) + (punctuation_noise * 0.30)
    return round(max(0.0, min(1.0, score)), 6)
