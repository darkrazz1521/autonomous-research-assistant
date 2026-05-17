"""Agentic workflow orchestration over the existing RAG pipeline."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AgenticPlan, ContradictionReport, EvidenceGraphReport, QueryUnderstandingResult, RAGAnswer, ReflectionReport, RetrievalLoopReport
from autonomous_research_assistant_data.rag.agentic.planner import AgenticPlanner
from autonomous_research_assistant_data.rag.agentic.retrieval_loop import AgenticRetrievalLoop
from autonomous_research_assistant_data.rag.evidence.evidence_graph import EvidenceGraphBuilder
from autonomous_research_assistant_data.rag.refinement.refiner import AnswerRefiner
from autonomous_research_assistant_data.rag.reflection.self_reflection import SelfReflectionEngine
from autonomous_research_assistant_data.rag.validation.contradiction_detector import ContradictionDetector


class AgenticResearchWorkflow:
    """Run a bounded planning, reflection, and refinement loop around RAG."""

    def __init__(self, config: AppConfig, retrieval_loop: AgenticRetrievalLoop) -> None:
        self.config = config
        self.planner = AgenticPlanner()
        self.retrieval_loop = retrieval_loop
        self.reflection = SelfReflectionEngine(config)
        self.evidence_graph = EvidenceGraphBuilder()
        self.contradictions = ContradictionDetector()
        self.refiner = AnswerRefiner()

    def build_plan(self, query: str, understanding: QueryUnderstandingResult) -> AgenticPlan:
        return self.planner.build_plan(query, understanding)

    def run_retrieval_loop(
        self,
        query: str,
        understanding: QueryUnderstandingResult,
        *,
        top_k: int,
        hybrid: bool,
        rerank: bool,
        context_window: bool,
    ) -> RetrievalLoopReport:
        _, report = self.retrieval_loop.run(
            query,
            understanding,
            top_k=top_k,
            hybrid=hybrid,
            rerank=rerank,
            context_window=context_window,
        )
        return report

    def postprocess(
        self,
        answer: RAGAnswer,
        *,
        plan: AgenticPlan,
        reflection_enabled: bool,
        refine_answer: bool,
        detect_contradictions: bool,
    ) -> tuple[RAGAnswer, ReflectionReport, EvidenceGraphReport, ContradictionReport]:
        reflection_report = self.reflection.inspect(answer, plan) if reflection_enabled else ReflectionReport()
        evidence_graph = self.evidence_graph.build(answer)
        contradiction_report = self.contradictions.detect(answer, evidence_graph) if detect_contradictions else ContradictionReport()
        refined = self.refiner.refine(answer, reflection_report, contradiction_report) if refine_answer else answer
        refined.retrieval_metadata["agentic_plan"] = plan.model_dump(mode="json")
        refined.retrieval_metadata["reflection_report"] = reflection_report.model_dump(mode="json")
        refined.retrieval_metadata["evidence_graph"] = evidence_graph.model_dump(mode="json")
        refined.retrieval_metadata["contradiction_report"] = contradiction_report.model_dump(mode="json")
        return refined, reflection_report, evidence_graph, contradiction_report
