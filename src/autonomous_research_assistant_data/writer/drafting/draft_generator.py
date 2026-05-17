"""Grounded long-form section drafting."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AnswerQualityReport, GroundingReport, RAGAnswer, RAGCitationRecord, RAGEvidenceChunk, RetrievalResult, SectionDraft, SectionEvidenceRecord, WritingSectionPlan, WriterSessionRecord
from autonomous_research_assistant_data.rag.answer_quality.scorer import AnswerQualityScorer
from autonomous_research_assistant_data.rag.grounding.grounding_checker import GroundingChecker
from autonomous_research_assistant_data.writer.citations.citation_manager import CitationManager
from autonomous_research_assistant_data.writer.style.style_controller import StyleController


class DraftGenerator:
    """Generate section drafts from grounded evidence rather than chunk concatenation."""

    def __init__(self, config: AppConfig, style_controller: StyleController | None = None) -> None:
        self.config = config
        self.style_controller = style_controller or StyleController()
        self.grounding = GroundingChecker(config)
        self.quality = AnswerQualityScorer(config)

    def _sentences(self, text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    def _best_sentences(self, results: list[RetrievalResult], keywords: list[str], limit: int) -> list[tuple[RetrievalResult, str]]:
        scored: list[tuple[int, float, RetrievalResult, str]] = []
        for result in results:
            for sentence in self._sentences(result.merged_context or result.chunk_text):
                lowered = sentence.lower()
                score = sum(keyword.lower() in lowered for keyword in keywords if keyword)
                scored.append((score, result.score, result, sentence))
        scored.sort(key=lambda item: (item[0], item[1], len(item[3])), reverse=True)
        picked: list[tuple[RetrievalResult, str]] = []
        seen: set[str] = set()
        for _, _, result, sentence in scored:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((result, sentence))
            if len(picked) >= limit:
                break
        return picked

    def _paragraphs(
        self,
        topic: str,
        plan: WritingSectionPlan,
        evidence: SectionEvidenceRecord,
        citation_manager: CitationManager,
        style_profile: dict[str, object],
        session: WriterSessionRecord | None,
    ) -> tuple[list[str], list[RAGCitationRecord], list[RAGEvidenceChunk]]:
        keywords = [topic, plan.title, *plan.required_terms[:6]]
        sentence_pairs = self._best_sentences(evidence.retrieval_results, keywords, limit=max(int(style_profile["paragraphs"]) * 2, 4))
        citations: list[RAGCitationRecord] = []
        evidence_chunks: list[RAGEvidenceChunk] = []
        paragraphs: list[str] = []
        transition = self.style_controller.transition(session.sections[-1].title if session and session.sections else None, plan.title)
        if sentence_pairs:
            lead_result, lead_sentence = sentence_pairs[0]
            lead_citation = citation_manager.build_citation_record(lead_result)
            citations.append(lead_citation)
            evidence_chunks.append(citation_manager.build_evidence_chunk(lead_result))
            paragraphs.append(
                f"{transition} {lead_sentence} {lead_citation.citation_label}"
            )
        detail_sentences = sentence_pairs[1:]
        sentence_limit = int(style_profile["sentence_limit"])
        bucket: list[str] = []
        bucket_citations: list[str] = []
        for result, sentence in detail_sentences:
            citation = citation_manager.build_citation_record(result)
            citations.append(citation)
            evidence_chunks.append(citation_manager.build_evidence_chunk(result))
            bucket.append(sentence)
            bucket_citations.append(citation.citation_label)
            if len(bucket) >= sentence_limit:
                paragraphs.append(" ".join(bucket + [bucket_citations[-1]]))
                bucket = []
                bucket_citations = []
        if bucket:
            paragraphs.append(" ".join(bucket + ([bucket_citations[-1]] if bucket_citations else [])))
        if evidence.coverage_score < self.config.writer.evidence_coverage_threshold:
            paragraphs.append(
                "The currently retrieved evidence does not fully cover every planned angle for this section, so the synthesis emphasizes the strongest grounded themes while preserving remaining gaps."
            )
        if evidence.contradiction_report and evidence.contradiction_report.contradiction_score > 0:
            paragraphs.append(
                "The retrieved corpus also includes partial disagreement or differing emphasis across sources, which should be interpreted as evidence heterogeneity rather than a resolved consensus."
            )
        return paragraphs[: int(style_profile["paragraphs"]) + 1], citation_manager.merge_duplicates(citations), evidence_chunks[: self.config.rag.synthesis.max_evidence_chunks]

    def _score(self, section_id: str, title: str, content: str, citations: list[RAGCitationRecord], evidence_chunks: list[RAGEvidenceChunk]) -> tuple[GroundingReport, AnswerQualityReport]:
        answer = RAGAnswer(query=title, answer=content, citations=citations, evidence_chunks=evidence_chunks)
        grounding = self.grounding.check(answer)
        quality = self.quality.score(answer, grounding)
        return grounding, quality

    def generate(
        self,
        topic: str,
        plan: WritingSectionPlan,
        evidence: SectionEvidenceRecord,
        *,
        style: str,
        depth: str,
        citation_manager: CitationManager,
        session: WriterSessionRecord | None = None,
    ) -> SectionDraft:
        profile = self.style_controller.resolve(style, depth)
        paragraphs, citations, evidence_chunks = self._paragraphs(topic, plan, evidence, citation_manager, profile, session)
        content = self.style_controller.normalize_text("\n\n".join(paragraphs))
        grounding, quality = self._score(plan.section_id, plan.title, content, citations, evidence_chunks)
        terminology = list(dict.fromkeys([term for term in plan.required_terms if term and len(term) > 2]))[:8]
        unresolved = []
        if evidence.coverage_score < self.config.writer.evidence_coverage_threshold:
            unresolved.append("evidence coverage remains partial")
        if grounding.unsupported_claim_ratio > (1.0 - self.config.writer.grounding_threshold):
            unresolved.append("some synthesized statements are weakly supported")
        summary = paragraphs[0][:280].strip() if paragraphs else plan.objective
        return SectionDraft(
            section_id=plan.section_id,
            title=plan.title,
            objective=plan.objective,
            content=content,
            summary=summary,
            citations=citations,
            evidence_chunks=evidence_chunks,
            grounding_report=grounding,
            answer_quality_report=quality,
            unresolved_gaps=unresolved,
            terminology=terminology,
            metadata={
                "coverage_score": evidence.coverage_score,
                "retrieval_query": evidence.query,
                "writing_decisions": ["grounded_synthesis", "section_level_evidence"],
            },
        )

