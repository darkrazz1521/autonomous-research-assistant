"""Evaluation framework for long-form writer outputs."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ResearchReport, WriterEvaluationReport


class WriterEvaluator:
    """Score report quality at the section and document level."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _avg(self, values: list[float]) -> float:
        return round(sum(values) / max(len(values), 1), 6)

    def evaluate(self, report: ResearchReport) -> WriterEvaluationReport:
        section_coherence = []
        grounding = []
        citation_correctness = []
        redundancy = []
        evidence_completeness = []
        synthesis_quality = []
        transition_quality = []
        term_sets = []
        for section in report.sections:
            if section.answer_quality_report:
                section_coherence.append(max(0.0, 1.0 - section.answer_quality_report.redundancy_score))
                redundancy.append(section.answer_quality_report.redundancy_score)
                synthesis_quality.append(section.answer_quality_report.synthesis_quality)
            if section.grounding_report:
                grounding.append(section.grounding_report.grounding_score)
                evidence_completeness.append(1.0 - section.grounding_report.unsupported_claim_ratio)
            citation_correctness.append(min(len(section.citations) / max(len(section.evidence_chunks), 1), 1.0))
            transition_quality.append(1.0 if "section" in section.content.lower() or "building on" in section.content.lower() else 0.7)
            term_sets.append(set(term.lower() for term in section.terminology))
        shared_terms = set.intersection(*term_sets) if term_sets else set()
        structural = min(len(report.sections) / max(self.config.writer.max_sections, 1), 1.0)
        evaluation = WriterEvaluationReport(
            evaluation_id=f"writer-{utc_now().strftime('%Y%m%d%H%M%S')}",
            session_id=report.session_id,
            topic=report.topic,
            report_type=report.report_type,
            section_coherence=self._avg(section_coherence),
            grounding_quality=self._avg(grounding),
            citation_correctness=self._avg(citation_correctness),
            redundancy_score=self._avg(redundancy),
            terminology_consistency=round(min(len(shared_terms) / max(len(set().union(*term_sets)) or 1, 1), 1.0), 6) if term_sets else 0.0,
            evidence_completeness=self._avg(evidence_completeness),
            structural_completeness=round(structural, 6),
            transition_quality=self._avg(transition_quality),
            synthesis_quality=self._avg(synthesis_quality),
            metadata={"section_count": len(report.sections)},
        )
        return evaluation

