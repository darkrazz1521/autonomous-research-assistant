"""Grounded long-form section drafting."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AnswerQualityReport, GroundingReport, QueryUnderstandingResult, RAGAnswer, RAGCitationRecord, RAGEvidenceChunk, SectionDraft, SectionEvidenceRecord, WritingSectionPlan, WriterSessionRecord
from autonomous_research_assistant_data.rag.answer_quality.scorer import AnswerQualityScorer
from autonomous_research_assistant_data.rag.grounding.grounding_checker import GroundingChecker
from autonomous_research_assistant_data.writer.citations.citation_manager import CitationManager
from autonomous_research_assistant_data.writer.style.style_controller import StyleController
from autonomous_research_assistant_data.writer.synthesis.semantic_synthesizer import SemanticSynthesizer


class DraftGenerator:
    """Generate section drafts from semantically merged grounded evidence."""

    def __init__(self, config: AppConfig, style_controller: StyleController | None = None) -> None:
        self.config = config
        self.style_controller = style_controller or StyleController()
        self.grounding = GroundingChecker(config)
        self.quality = AnswerQualityScorer(config)
        self.synthesizer = SemanticSynthesizer()

    def _score(self, title: str, content: str, citations: list[RAGCitationRecord], evidence_chunks: list[RAGEvidenceChunk]) -> tuple[GroundingReport, AnswerQualityReport]:
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
        understanding: QueryUnderstandingResult,
        citation_manager: CitationManager,
        session: WriterSessionRecord | None = None,
        quality_controls: dict[str, bool] | None = None,
    ) -> SectionDraft:
        quality_controls = quality_controls or {}
        profile = self.style_controller.resolve(style, depth)
        rhetorical_plan = dict(plan.metadata.get("rhetorical_plan", {})) if quality_controls.get("rhetorical_planning") else dict(plan.metadata.get("rhetorical_plan", {}))
        synthesis = self.synthesizer.synthesize(
            topic=topic,
            plan=plan,
            understanding=understanding,
            results=evidence.retrieval_results,
            citation_manager=citation_manager,
            style_profile=profile,
            previous_title=session.sections[-1].title if session and session.sections else None,
            prior_claims=list(session.metadata.get("claim_memory", [])) if session and quality_controls.get("claim_dedup") else [],
            rhetorical_plan=rhetorical_plan,
        )
        paragraphs = list(synthesis["paragraphs"])
        cited_results = list(synthesis["cited_results"])
        citations: list[RAGCitationRecord] = []
        evidence_chunks: list[RAGEvidenceChunk] = []
        for result in cited_results:
            citations.append(citation_manager.build_citation_record(result))
            evidence_chunks.append(citation_manager.build_evidence_chunk(result))
        citations = citation_manager.merge_duplicates(citations)
        for index, paragraph in enumerate(paragraphs):
            updated_paragraph, paragraph_citations = citation_manager.attach_sentence_citation(paragraph, cited_results)
            paragraphs[index] = updated_paragraph
            citations.extend(paragraph_citations)
        citations = citation_manager.merge_duplicates(citations)
        content = self.style_controller.normalize_text("\n\n".join(paragraphs))
        grounding, quality = self._score(plan.title, content, citations, evidence_chunks[: self.config.rag.synthesis.max_evidence_chunks])
        terminology = list(dict.fromkeys([term for term in plan.required_terms if term and len(term) > 2]))[:8]
        unresolved = []
        if evidence.coverage_score < self.config.writer.evidence_coverage_threshold:
            unresolved.append("evidence coverage remains partial")
        if grounding.unsupported_claim_ratio > (1.0 - self.config.writer.grounding_threshold):
            unresolved.append("some synthesized statements remain weakly supported")
        if evidence.contradiction_report and evidence.contradiction_report.contradiction_score > 0:
            unresolved.append("cross-source evidence requires careful interpretation")
        summary = paragraphs[0][:280].strip() if paragraphs else plan.objective
        discourse_trace = self.style_controller.discourse_trace(
            session.sections[-1].title if session and session.sections else None,
            plan.title,
            rhetorical_role="comparison" if "comparison" in plan.title.lower() else "section_opening",
        )
        return SectionDraft(
            section_id=plan.section_id,
            title=plan.title,
            objective=plan.objective,
            content=content,
            summary=summary,
            citations=citations,
            evidence_chunks=evidence_chunks[: self.config.rag.synthesis.max_evidence_chunks],
            grounding_report=grounding,
            answer_quality_report=quality,
            unresolved_gaps=unresolved,
            terminology=terminology,
            metadata={
                "coverage_score": evidence.coverage_score,
                "retrieval_query": evidence.query,
                "writing_decisions": [
                    "semantic_synthesis",
                    "claim_level_evidence_aggregation",
                    "citation_grounded_paragraphs",
                    *[name for name, enabled in quality_controls.items() if enabled],
                ],
                "synthesis_trace": {
                    "claim_count": len(synthesis["claims"]),
                    "cluster_count": len(synthesis["clusters"]),
                    "paragraph_trace": synthesis["traces"],
                    "claim_graph_summary": synthesis.get("claim_graph_summary", {}),
                },
                "comparison_reasoning": [claim["claim_type"] for claim in synthesis["claims"] if claim["claim_type"] == "comparison"],
                "claim_texts": [str(claim.get("claim_text", "")) for claim in synthesis["claims"]],
                "claim_graph_summary": synthesis.get("claim_graph_summary", {}),
                "normalization_reports": synthesis.get("normalization_reports", []),
                "discourse_trace": discourse_trace,
                "rhetorical_plan": rhetorical_plan,
                "paraphrase_trace": [dict(claim.get("paraphrase_meta", {})) for claim in synthesis["claims"]],
            },
        )
