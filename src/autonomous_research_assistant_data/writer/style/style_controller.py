"""Scientific tone and style controls for long-form report writing."""

from __future__ import annotations

import re
from typing import Any


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

    def resolve(self, style: str, depth: str = "standard") -> dict[str, Any]:
        base = dict(STYLE_PROFILES.get(style, STYLE_PROFILES["technical"]))
        multiplier = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["standard"])
        base["depth"] = depth
        base["style"] = style
        base["paragraphs"] = max(2, int(round(base["paragraphs"] * multiplier)))
        base["sentence_limit"] = max(2, int(round(base["sentence_limit"] * max(multiplier, 0.9))))
        return base

    def transition(self, previous_title: str | None, current_title: str) -> str:
        if not previous_title:
            return f"This section examines {current_title.lower()} through the retrieved evidence."
        return (
            f"Building on the preceding discussion of {previous_title.lower()}, "
            f"this section focuses on {current_title.lower()}."
        )

    def normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def section_heading(self, title: str, level: int = 2) -> str:
        return f'{"#" * max(level, 2)} {title}'

