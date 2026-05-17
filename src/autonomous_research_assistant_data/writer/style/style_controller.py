"""Scientific tone and style controls for long-form report writing."""

from __future__ import annotations

import re
from typing import Any

from autonomous_research_assistant_data.writer.style.discourse_variation import DiscourseVariationEngine
from autonomous_research_assistant_data.writer.style.discourse_controller import DiscourseController


STYLE_PROFILES: dict[str, dict[str, Any]] = {
    "technical": {"paragraphs": 3, "sentence_limit": 3, "depth_multiplier": 1.0, "citation_density": 0.9},
    "survey": {"paragraphs": 4, "sentence_limit": 3, "depth_multiplier": 1.2, "citation_density": 1.0},
    "academic": {"paragraphs": 3, "sentence_limit": 4, "depth_multiplier": 1.1, "citation_density": 1.0},
    "comparison": {"paragraphs": 4, "sentence_limit": 3, "depth_multiplier": 1.1, "citation_density": 0.95},
    "tutorial": {"paragraphs": 3, "sentence_limit": 3, "depth_multiplier": 0.95, "citation_density": 0.8},
    "benchmark": {"paragraphs": 4, "sentence_limit": 3, "depth_multiplier": 1.15, "citation_density": 1.0},
}

DEPTH_PROFILES: dict[str, float] = {"brief": 0.8, "standard": 1.0, "deep": 1.25}


class StyleController:
    """Resolve style-specific writing parameters and sentence shaping."""

    def __init__(self) -> None:
        self.discourse = DiscourseController()
        self.variation = DiscourseVariationEngine()

    def resolve(self, style: str, depth: str = "standard") -> dict[str, Any]:
        base = dict(STYLE_PROFILES.get(style, STYLE_PROFILES["technical"]))
        multiplier = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["standard"])
        base["depth"] = depth
        base["style"] = style
        base["paragraphs"] = max(2, int(round(base["paragraphs"] * multiplier)))
        base["sentence_limit"] = max(2, int(round(base["sentence_limit"] * max(multiplier, 0.9))))
        base["rhetorical_flow"] = self.rhetorical_flow(style)
        return base

    def rhetorical_flow(self, style: str) -> list[str]:
        if style == "comparison":
            return ["claim", "difference", "evidence", "interpretation"]
        if style in {"survey", "academic"}:
            return ["context", "claim", "evidence", "interpretation"]
        return ["claim", "evidence", "interpretation"]

    def transition(self, previous_title: str | None, current_title: str, *, rhetorical_role: str = "section_opening") -> str:
        current = current_title.lower()
        if not previous_title:
            if rhetorical_role == "fallback":
                return f"The retrieved corpus provides only limited support for {current}."
            transition, _ = self.variation.transition(previous_title=None, current_title=current_title, rhetorical_role=rhetorical_role)
            return transition
        previous = previous_title.lower()
        if rhetorical_role == "fallback":
            return f"Relative to the preceding discussion of {previous}, the present section on {current} remains only partially supported by the retrieved evidence."
        transition, _ = self.variation.transition(previous_title=previous_title, current_title=current_title, rhetorical_role=rhetorical_role)
        return transition

    def discourse_trace(self, previous_title: str | None, current_title: str, *, rhetorical_role: str = "section_opening") -> dict[str, object]:
        _, trace = self.variation.transition(previous_title=previous_title, current_title=current_title, rhetorical_role=rhetorical_role)
        trace["recent_patterns"] = self.variation.recent_patterns()
        return trace

    def connector(self, current_title: str) -> str:
        return self.variation.connector(current_title)

    def diversify_sentence(self, sentence: str, current_title: str, index: int) -> str:
        return self.variation.diversify_sentence_opening(sentence, current_title, index)

    def evidence_phrase(self, section_name: object, canonical_label: object) -> str:
        label = str(canonical_label or section_name or "the retrieved section").lower()
        mapping = {
            "abstract": "abstract-level framing",
            "introduction": "introductory discussion",
            "methodology": "methodological description",
            "methods": "methodological description",
            "results": "reported results",
            "discussion": "discussion sections",
            "conclusion": "concluding interpretation",
            "related_work": "related-work synthesis",
            "experiments": "experimental evidence",
            "preliminaries": "preliminary formulation",
        }
        return mapping.get(label, f"the {str(section_name).lower()} section")

    def normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def section_heading(self, title: str, level: int = 2) -> str:
        return f'{"#" * max(level, 2)} {title}'

    def introduction(self, topic: str, report_type: str, section_titles: list[str]) -> str:
        if section_titles:
            return (
                f"This {report_type.replace('-', ' ')} synthesizes grounded scientific evidence on {topic}. "
                f"It is organized around {', '.join(title.lower() for title in section_titles[:3])}, with emphasis on evidence-backed interpretation rather than raw retrieval traces."
            )
        return f"This report synthesizes grounded scientific evidence on {topic}."

    def conclusion(self, unresolved: list[str], summaries: list[str]) -> str:
        body = " ".join(summary.strip() for summary in summaries if summary.strip())
        if not body:
            body = "The report consolidates the strongest retrieved evidence while preserving uncertainty where the literature remains incomplete."
        if unresolved:
            body += f" Remaining open issues include {', '.join(unresolved[:3])}."
        return body.strip()
