"""Hallucination and contradiction heuristics."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import GroundingReport, RAGAnswer


class HallucinationDetector:
    """Estimate hallucination risk from grounding signals."""

    def detect(self, answer: RAGAnswer, grounding: GroundingReport) -> dict[str, float]:
        citation_sparse = 1.0 - min(len(answer.citations) / max(len(answer.evidence_chunks), 1), 1.0)
        hallucination_probability = min(
            1.0,
            (grounding.unsupported_claim_ratio * 0.55)
            + ((1.0 - grounding.citation_coverage) * 0.25)
            + (citation_sparse * 0.20),
        )
        return {
            "grounding_score": grounding.grounding_score,
            "unsupported_claim_ratio": grounding.unsupported_claim_ratio,
            "hallucination_probability": round(hallucination_probability, 6),
            "citation_coverage": grounding.citation_coverage,
            "evidence_density": grounding.evidence_density,
        }
