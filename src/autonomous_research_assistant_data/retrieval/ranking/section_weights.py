"""Section-aware retrieval weighting."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig


def resolve_section_weight(config: AppConfig, canonical_label: str | None, *, enabled: bool = True) -> float:
    if not enabled or not config.retrieval.section_weights.enabled:
        return 1.0
    label = (canonical_label or "").strip().lower()
    if not label:
        return config.retrieval.section_weights.default_weight
    return float(config.retrieval.section_weights.weights.get(label, config.retrieval.section_weights.default_weight))
