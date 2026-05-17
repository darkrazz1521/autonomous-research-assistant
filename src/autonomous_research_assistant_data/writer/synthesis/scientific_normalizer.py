"""Normalize scientific artifacts such as equations and table fragments into prose-friendly summaries."""

from __future__ import annotations

import re


class ScientificNormalizer:
    """Detect and verbalize artifact-heavy scientific fragments."""

    EQUATION_PATTERN = re.compile(r"(?:[A-Za-z_]+\([^)]*\)|[A-Za-z][A-Za-z0-9_]*\s*=\s*[^.]{4,}|\\[a-zA-Z]+|theta|lambda|kl)", re.IGNORECASE)
    TABLE_PATTERN = re.compile(r"\b(?:table|row|column|header|metric|dataset)\b", re.IGNORECASE)

    def detect_equation(self, text: str) -> bool:
        return bool(self.EQUATION_PATTERN.search(text))

    def detect_table_fragment(self, text: str) -> bool:
        digit_ratio = len(re.findall(r"\d", text)) / max(len(text), 1)
        return bool(self.TABLE_PATTERN.search(text)) or digit_ratio > 0.22

    def summarize_equation(self, text: str) -> str:
        lowered = text.lower()
        if "grpo" in lowered and any(token in lowered for token in ("kl", "objective", "loss", "clip")):
            return "The GRPO formulation describes a clipped policy optimization objective with regularization to stabilize updates."
        if "ppo" in lowered and any(token in lowered for token in ("objective", "clip", "surrogate")):
            return "The PPO formulation centers on a clipped surrogate objective designed to limit destabilizing policy changes."
        if any(token in lowered for token in ("loss", "objective", "optimiz")):
            return "The retrieved passage describes an optimization objective rather than a narrative scientific claim."
        return "The retrieved fragment is primarily mathematical and is better interpreted as a formal specification than direct prose."

    def summarize_table(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("benchmark", "dataset", "accuracy", "reward", "score")):
            return "The reported table summarizes comparative empirical results across datasets, metrics, or training conditions."
        return "The retrieved fragment appears to summarize tabular empirical evidence rather than a standalone narrative statement."

    def normalize_fragment(self, text: str) -> tuple[str, dict[str, object]]:
        cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        report = {
            "equation_detected": False,
            "table_fragment_detected": False,
            "malformed_fragment_suppressed": False,
        }
        if not cleaned:
            report["malformed_fragment_suppressed"] = True
            return "", report
        if self.detect_equation(cleaned):
            report["equation_detected"] = True
            return self.summarize_equation(cleaned), report
        if self.detect_table_fragment(cleaned):
            report["table_fragment_detected"] = True
            return self.summarize_table(cleaned), report
        if cleaned.count("|") >= 2 or cleaned.count("...") >= 2:
            report["malformed_fragment_suppressed"] = True
            return "", report
        return cleaned, report
