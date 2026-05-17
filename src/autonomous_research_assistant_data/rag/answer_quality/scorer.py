"""Answer quality scoring for grounded RAG responses."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AnswerQualityReport, GroundingReport, RAGAnswer


class AnswerQualityScorer:
    """Compute overall answer quality metrics from answer and grounding signals."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def score(self, answer: RAGAnswer, grounding: GroundingReport) -> AnswerQualityReport:
        text = answer.answer
        tokens = re.findall(r"\w+", text)
        unique_ratio = len(set(token.lower() for token in tokens)) / max(len(tokens), 1)
        citation_density = len(answer.citations) / max(len(tokens), 1)
        semantic_completeness = min(len(tokens) / 120, 1.0)
        redundancy_score = max(0.0, 1.0 - unique_ratio)
        factual_consistency = max(0.0, 1.0 - grounding.unsupported_claim_ratio)
        retrieval_alignment = grounding.grounding_score
        synthesis_quality = (semantic_completeness * 0.4) + ((1.0 - redundancy_score) * 0.2) + (citation_density * 10 * 0.2) + (factual_consistency * 0.2)
        overall = (
            citation_density * 6 * 0.1
            + grounding.grounding_score * 0.30
            + semantic_completeness * 0.15
            + (1.0 - redundancy_score) * self.config.rag.answer_quality.redundancy_penalty_weight
            + factual_consistency * self.config.rag.answer_quality.factual_consistency_weight
            + retrieval_alignment * self.config.rag.answer_quality.retrieval_alignment_weight
            + synthesis_quality * 0.05
        )
        return AnswerQualityReport(
            citation_density=round(citation_density, 6),
            grounding_score=grounding.grounding_score,
            semantic_completeness=round(semantic_completeness, 6),
            redundancy_score=round(redundancy_score, 6),
            factual_consistency=round(factual_consistency, 6),
            retrieval_alignment=round(retrieval_alignment, 6),
            synthesis_quality=round(min(synthesis_quality, 1.0), 6),
            overall_answer_quality_score=round(min(overall, 1.0), 6),
        )
