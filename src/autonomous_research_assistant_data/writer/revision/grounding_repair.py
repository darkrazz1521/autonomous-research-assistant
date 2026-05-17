"""Repair weakly grounded prose by removing unsupported sentences."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import GroundingReport


class GroundingRepair:
    """Drop or compress unsupported claims to improve groundedness."""

    def repair(self, text: str, grounding: GroundingReport | None) -> tuple[str, dict[str, object]]:
        if grounding is None or not grounding.unsupported_claims:
            return text, {"removed_unsupported_claims": 0}
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        removed = 0
        normalized_unsupported = {re.sub(r"\s+", " ", claim.lower()).strip() for claim in grounding.unsupported_claims}
        repaired: list[str] = []
        for paragraph in paragraphs:
            sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", paragraph) if segment.strip()]
            kept = []
            for sentence in sentences:
                normalized = re.sub(r"\s+", " ", sentence.lower()).strip()
                if normalized in normalized_unsupported:
                    removed += 1
                    continue
                kept.append(sentence)
            if kept:
                repaired.append(" ".join(kept))
        return "\n\n".join(repaired).strip(), {"removed_unsupported_claims": removed}

