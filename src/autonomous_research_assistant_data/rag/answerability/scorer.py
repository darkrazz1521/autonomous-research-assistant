"""Score which chunks are most likely to directly answer a query."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult


class AnswerabilityScorer:
    """Estimate direct-answer likelihood before prompt assembly."""

    def score(self, result: RetrievalResult, understanding: QueryUnderstandingResult) -> float:
        text = (result.chunk_text or "").lower()
        section = (result.section_name or "").lower()
        query_terms = set(re.findall(r"[a-z0-9][a-z0-9\-]+", understanding.normalized_query.lower()))
        overlap = len(query_terms.intersection(set(re.findall(r"[a-z0-9][a-z0-9\-]+", text)))) / max(len(query_terms), 1)
        definition_bonus = 0.0
        if understanding.query_type == "definition" and any(token in text for token in (" is ", " refers to ", " framework", " optimization")):
            definition_bonus += 0.12
        comparison_bonus = 0.0
        if understanding.query_type == "comparison" and any(token in text for token in ("however", "whereas", "compared", "in contrast")):
            comparison_bonus += 0.10
        method_bonus = 0.0
        if understanding.query_type == "methodology_explanation" and section in {"methods", "methodology"}:
            method_bonus += 0.10
        entity_bonus = 0.0
        if understanding.entities and sum(entity.lower() in text for entity in understanding.entities[:3]) >= 1:
            entity_bonus += 0.08
        section_bonus = 0.0
        if understanding.query_type == "definition" and section in {"abstract", "introduction"}:
            section_bonus += 0.08
        if understanding.query_type in {"comparison", "benchmark_performance"} and section in {"results", "discussion", "experiments"}:
            section_bonus += 0.08
        return round(min(1.0, overlap + definition_bonus + comparison_bonus + method_bonus + entity_bonus + section_bonus), 6)

    def rank(self, results: list[RetrievalResult], understanding: QueryUnderstandingResult) -> list[RetrievalResult]:
        ranked: list[RetrievalResult] = []
        for result in results:
            updated = result.model_copy(deep=True)
            score = self.score(updated, understanding)
            updated.final_score_breakdown["answerability_score"] = score
            updated.score = round(updated.score + score, 6)
            ranked.append(updated)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked
