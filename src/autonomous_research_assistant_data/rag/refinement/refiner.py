"""Multi-pass answer refinement."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import ContradictionReport, RAGAnswer, ReflectionReport


class AnswerRefiner:
    """Reduce repetition and surface uncertainty in grounded answers."""

    def refine(self, answer: RAGAnswer, reflection: ReflectionReport, contradiction: ContradictionReport) -> RAGAnswer:
        refined = answer.model_copy(deep=True)
        paragraphs = [segment.strip() for segment in refined.answer.split("\n\n") if segment.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for paragraph in paragraphs:
            normalized = re.sub(r"\s+", " ", paragraph.lower())
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(paragraph)
        if contradiction.contradiction_score > 0:
            deduped.append("Uncertainty: Some retrieved evidence may reflect different emphases or partially conflicting interpretations across sources.")
        refined.answer = "\n\n".join(deduped)
        refined.retrieval_metadata["refinement_report"] = {
            "removed_duplicates": max(len(paragraphs) - len(deduped), 0),
            "reflection_actions": reflection.refinement_actions,
            "contradiction_score": contradiction.contradiction_score,
        }
        return refined
