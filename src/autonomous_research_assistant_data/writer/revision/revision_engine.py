"""Iterative refinement for long-form section drafts."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AnswerQualityReport, GroundingReport, RAGAnswer, SectionDraft, SectionEvidenceRecord, WriterRevisionRecord
from autonomous_research_assistant_data.rag.answer_quality.scorer import AnswerQualityScorer
from autonomous_research_assistant_data.rag.grounding.grounding_checker import GroundingChecker


class RevisionEngine:
    """Run bounded revisions to improve coherence, grounding, and citation alignment."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.grounding = GroundingChecker(config)
        self.quality = AnswerQualityScorer(config)

    def _score(self, draft: SectionDraft) -> tuple[GroundingReport, AnswerQualityReport]:
        answer = RAGAnswer(query=draft.title, answer=draft.content, citations=draft.citations, evidence_chunks=draft.evidence_chunks)
        grounding = self.grounding.check(answer)
        quality = self.quality.score(answer, grounding)
        return grounding, quality

    def _dedupe(self, text: str) -> tuple[str, int]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        removed = 0
        for paragraph in paragraphs:
            normalized = re.sub(r"\s+", " ", paragraph.lower())
            if normalized in seen:
                removed += 1
                continue
            seen.add(normalized)
            deduped.append(paragraph)
        return "\n\n".join(deduped), removed

    def revise(self, draft: SectionDraft, evidence: SectionEvidenceRecord, *, passes: int) -> SectionDraft:
        revised = draft.model_copy(deep=True)
        base_quality = revised.answer_quality_report.overall_answer_quality_score if revised.answer_quality_report else 0.0
        for revision_index in range(1, min(passes, self.config.writer.max_revision_passes) + 1):
            content, removed = self._dedupe(revised.content)
            actions = ["deduplicate_paragraphs"] if removed else []
            if evidence.contradiction_report and evidence.contradiction_report.contradiction_score > 0 and "Uncertainty:" not in content:
                content += "\n\nUncertainty: Some retrieved sources place different emphasis on this topic, so the section preserves that variation rather than forcing a false consensus."
                actions.append("surface_uncertainty")
            if evidence.coverage_score < self.config.writer.evidence_coverage_threshold and "remaining evidence gaps" not in content.lower():
                content += "\n\nThis section should be read with remaining evidence gaps in mind because some planned subtopics were only partially covered by the retrieved corpus."
                actions.append("mark_evidence_gaps")
            revised.content = content.strip()
            grounding, quality = self._score(revised)
            revised.grounding_report = grounding
            revised.answer_quality_report = quality
            refinement_gain = round(quality.overall_answer_quality_score - base_quality, 6)
            coherence = round(max(0.0, 1.0 - grounding.unsupported_claim_ratio - (quality.redundancy_score * 0.5)), 6)
            revised.revision_history.append(
                WriterRevisionRecord(
                    revision_index=revision_index,
                    actions=actions or ["stability_pass"],
                    refinement_gain=refinement_gain,
                    unsupported_claim_frequency=grounding.unsupported_claim_ratio,
                    redundancy_score=quality.redundancy_score,
                    coherence_score=coherence,
                    metadata={"coverage_score": evidence.coverage_score},
                )
            )
            base_quality = quality.overall_answer_quality_score
        return revised

