"""Grounding verification for generated answers."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import GroundingReport, RAGAnswer


class GroundingChecker:
    """Check whether generated answers remain supported by retrieved evidence."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _token_overlap(self, claim: str, evidence_text: str) -> float:
        tokens = [token for token in re.findall(r"[a-z0-9][a-z0-9\-]+", claim.lower()) if len(token) > 3]
        if not tokens:
            return 0.0
        overlap = sum(token in evidence_text for token in set(tokens))
        return overlap / max(len(set(tokens)), 1)

    def check(self, answer: RAGAnswer) -> GroundingReport:
        evidence_text = " ".join(chunk.quote for chunk in answer.evidence_chunks).lower()
        claims = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer.answer) if segment.strip()]
        unsupported: list[str] = []
        supported = 0
        contradiction_signals = 0
        for claim in claims:
            overlap_ratio = self._token_overlap(claim, evidence_text)
            if overlap_ratio >= 0.35:
                supported += 1
            else:
                unsupported.append(claim)
            lowered = claim.lower()
            if any(marker in lowered for marker in ("however", "in contrast", "whereas", "but")):
                contradiction_signals += 1
        citation_coverage = min(len(answer.citations) / max(len(claims), 1), 1.0)
        grounding_score = ((supported / max(len(claims), 1)) * 0.55) + (citation_coverage * 0.25) + (min(len(answer.evidence_chunks) / max(len(claims), 1), 1.0) * 0.20)
        hallucination_probability = max(0.0, 1.0 - grounding_score)
        warnings: list[str] = []
        if grounding_score < self.config.rag.grounding.low_confidence_warning_threshold:
            warnings.append("Grounding score is below the low-confidence threshold.")
        if unsupported:
            warnings.append("Some answer sentences could not be aligned to retrieved evidence.")
        return GroundingReport(
            grounding_score=round(grounding_score, 6),
            unsupported_claim_ratio=round(len(unsupported) / max(len(claims), 1), 6),
            hallucination_probability=round(hallucination_probability, 6),
            citation_coverage=round(citation_coverage, 6),
            evidence_density=round(sum(len(chunk.quote.split()) for chunk in answer.evidence_chunks) / max(len(claims), 1), 6),
            contradiction_score=round(min(contradiction_signals / max(len(claims), 1), 1.0), 6),
            low_confidence_warnings=warnings,
            unsupported_claims=unsupported,
            metadata={
                "supported_claim_count": supported,
                "claim_count": len(claims),
            },
        )
