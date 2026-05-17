"""End-to-end Phase 6 single-agent writing workflow."""

from __future__ import annotations

import hashlib

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, WriterObservabilityReport
from autonomous_research_assistant_data.rag.query_understanding.understanding import QueryUnderstandingAnalyzer
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi
from autonomous_research_assistant_data.storage.file_store import append_jsonl, write_json
from autonomous_research_assistant_data.writer.citations.citation_manager import CitationManager
from autonomous_research_assistant_data.writer.drafting.draft_generator import DraftGenerator
from autonomous_research_assistant_data.writer.evaluation.writer_evaluator import WriterEvaluator
from autonomous_research_assistant_data.writer.memory.writing_memory import WritingMemoryStore
from autonomous_research_assistant_data.writer.orchestration.section_orchestrator import SectionOrchestrator
from autonomous_research_assistant_data.writer.outlining.outline_planner import OutlinePlanner
from autonomous_research_assistant_data.writer.planner.writing_planner import WritingPlanner
from autonomous_research_assistant_data.writer.report_generation.report_builder import ReportBuilder
from autonomous_research_assistant_data.writer.revision.revision_engine import RevisionEngine
from autonomous_research_assistant_data.writer.style.style_controller import StyleController
from autonomous_research_assistant_data.writer.synthesis.report_synthesizer import ReportSynthesizer


class WritingWorkflow:
    """Coordinate planning, retrieval, drafting, revision, synthesis, evaluation, and persistence."""

    def __init__(self, config: AppConfig, retrieval_api: RetrievalApi | None = None) -> None:
        self.config = config
        self.retrieval_api = retrieval_api or RetrievalApi(config)
        self.query_understanding = QueryUnderstandingAnalyzer(config)
        self.outline_planner = OutlinePlanner(config)
        self.writing_planner = WritingPlanner(config)
        self.memory = WritingMemoryStore(config)
        self.section_orchestrator = SectionOrchestrator(config, self.retrieval_api)
        self.style_controller = StyleController()
        self.draft_generator = DraftGenerator(config, self.style_controller)
        self.revision_engine = RevisionEngine(config)
        self.report_synthesizer = ReportSynthesizer(config, self.style_controller)
        self.report_builder = ReportBuilder(config)
        self.evaluator = WriterEvaluator(config)

    def _session_id(self, topic: str, report_type: str) -> str:
        return hashlib.sha256(f"{topic}:{report_type}".encode("utf-8")).hexdigest()[:16]

    def _write_observability(self, session, report) -> WriterObservabilityReport:
        observability = WriterObservabilityReport(
            session_id=session.session_id,
            topic=session.topic,
            section_quality={
                section.section_id: float(section.answer_quality_report.overall_answer_quality_score if section.answer_quality_report else 0.0)
                for section in report.sections
            },
            evidence_coverage={section.section_id: float(section.metadata.get("coverage_score", 0.0)) for section in report.sections},
            revision_gains={
                section.section_id: float(section.revision_history[-1].refinement_gain if section.revision_history else 0.0)
                for section in report.sections
            },
            citation_density={
                section.section_id: float(section.answer_quality_report.citation_density if section.answer_quality_report else 0.0)
                for section in report.sections
            },
            unsupported_claim_frequency={
                section.section_id: float(section.grounding_report.unsupported_claim_ratio if section.grounding_report else 0.0)
                for section in report.sections
            },
            redundancy_ratios={
                section.section_id: float(section.answer_quality_report.redundancy_score if section.answer_quality_report else 0.0)
                for section in report.sections
            },
            writing_coherence={
                section.section_id: float(section.revision_history[-1].coherence_score if section.revision_history else 0.0)
                for section in report.sections
            },
            retrieval_usage_per_section={section.section_id: len(section.evidence_chunks) for section in report.sections},
            grounding_quality_per_section={
                section.section_id: float(section.grounding_report.grounding_score if section.grounding_report else 0.0)
                for section in report.sections
            },
            metadata={"generated_at": utc_now(), "report_id": report.report_id},
        )
        write_json(self.config.writer.writer_observability_dir / "latest_writer_observability.json", observability.model_dump(mode="json"))
        append_jsonl(self.config.writer.writer_observability_dir / "writer_observability_log.jsonl", observability.model_dump(mode="json"))
        return observability

    def run(
        self,
        *,
        topic: str,
        report_type: str,
        style: str | None = None,
        depth: str = "standard",
        max_sections: int | None = None,
        revision_passes: int | None = None,
        citation_style: str | None = None,
        export_format: str | None = None,
    ):
        understanding: QueryUnderstandingResult = self.query_understanding.analyze(topic)
        outline = self.outline_planner.build(topic, report_type, understanding, max_sections=max_sections)
        plan = self.writing_planner.build(topic, report_type, outline, understanding)
        session_id = self._session_id(topic, report_type)
        session = self.memory.load(session_id) or self.memory.create(
            session_id=session_id,
            topic=topic,
            report_type=report_type,
            title=outline.title,
        )
        session = self.memory.attach_outline(session, outline)
        session = self.memory.attach_plan(session, plan)
        citation_manager = CitationManager(citation_style or self.config.writer.citation_style)
        final_sections = []
        for section_plan in plan.section_sequence[: self.config.writer.max_sections]:
            evidence = self.section_orchestrator.orchestrate(topic, section_plan, understanding, session=session)
            draft = self.draft_generator.generate(
                topic,
                section_plan,
                evidence,
                style=style or self.config.writer.style,
                depth=depth,
                citation_manager=citation_manager,
                session=session,
            )
            revised = self.revision_engine.revise(draft, evidence, passes=revision_passes or self.config.writer.max_revision_passes)
            final_sections.append(revised)
            session = self.memory.add_section(session, revised)
        report_id = f"report-{utc_now().strftime('%Y%m%d%H%M%S')}"
        bibliography = citation_manager.bibliography([citation for section in final_sections for citation in section.citations])
        report = self.report_synthesizer.synthesize(
            report_id=report_id,
            session_id=session.session_id,
            topic=topic,
            report_type=report_type,
            style=style or self.config.writer.style,
            citation_style=citation_style or self.config.writer.citation_style,
            export_format=export_format or self.config.writer.export_format,
            outline=outline,
            sections=final_sections,
            bibliography=bibliography,
            session=session,
        )
        evaluation = self.evaluator.evaluate(report)
        report.evaluation = evaluation
        artifact_paths = self.report_builder.persist(report)
        report.metadata["artifact_paths"] = artifact_paths
        self.report_builder.persist(report)
        self.memory.save(session)
        self._write_observability(session, report)
        write_json(self.config.writer.writer_evaluation_dir / f"{evaluation.evaluation_id}.json", evaluation.model_dump(mode="json"))
        return report
