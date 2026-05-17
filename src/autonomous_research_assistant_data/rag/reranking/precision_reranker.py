"""Enhanced reranking with intent alignment and answerability signals."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult
from autonomous_research_assistant_data.rag.answerability.scorer import AnswerabilityScorer


class PrecisionReranker:
    """Apply a precision-oriented reranking layer over retrieved chunks."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.answerability = AnswerabilityScorer()

    def rerank(self, results: list[RetrievalResult], understanding: QueryUnderstandingResult) -> list[RetrievalResult]:
        reranked: list[RetrievalResult] = []
        for result in results:
            updated = result.model_copy(deep=True)
            answerability_score = self.answerability.score(updated, understanding)
            citation_bonus = min(float(updated.metadata.get("citation_density", 0.0)) * 10, 1.0) * self.config.rag.reranking.citation_density_weight
            section_bonus = 0.0
            if understanding.query_type in {"definition", "methodology_explanation"} and (updated.canonical_section_label or "") in {"abstract", "introduction", "methodology", "methods"}:
                section_bonus = self.config.rag.reranking.section_relevance_weight
            if understanding.query_type in {"comparison", "benchmark_performance"} and (updated.canonical_section_label or "") in {"results", "discussion"}:
                section_bonus = self.config.rag.reranking.section_relevance_weight
            intent_bonus = 0.0
            text = (updated.chunk_text + " " + updated.section_name).lower()
            if any(topic.lower() in text for topic in understanding.target_topics[:6]):
                intent_bonus = self.config.rag.reranking.semantic_intent_weight
            updated.score = round(
                updated.score
                + answerability_score * self.config.rag.reranking.answerability_weight
                + citation_bonus
                + section_bonus
                + intent_bonus,
                6,
            )
            updated.final_score_breakdown["precision_answerability"] = answerability_score
            updated.final_score_breakdown["precision_citation_bonus"] = citation_bonus
            updated.final_score_breakdown["precision_section_bonus"] = section_bonus
            updated.final_score_breakdown["precision_intent_bonus"] = intent_bonus
            reranked.append(updated)
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked
