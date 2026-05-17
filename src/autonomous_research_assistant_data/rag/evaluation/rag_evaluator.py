"""Evaluation framework for Phase 5 RAG."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import RAGEvaluationReport
from autonomous_research_assistant_data.retrieval.evaluation.framework import RetrievalEvaluationFramework
from autonomous_research_assistant_data.rag.orchestration.rag_pipeline import RAGPipeline
from autonomous_research_assistant_data.storage.file_store import write_json


class RAGEvaluator:
    """Run lightweight end-to-end RAG evaluation on retrieval probes."""

    def __init__(self, config: AppConfig, pipeline: RAGPipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self.retrieval_eval = RetrievalEvaluationFramework(config, pipeline.retrieval_api)

    def evaluate(self, *, probe_limit: int | None = None) -> RAGEvaluationReport:
        probes = self.retrieval_eval.build_manual_probes(limit=probe_limit or self.config.retrieval.evaluation.default_probe_count)
        grounding_scores: list[float] = []
        citation_scores: list[float] = []
        relevance_scores: list[float] = []
        completeness_scores: list[float] = []
        hallucination_scores: list[float] = []
        latency_scores: list[float] = []
        synthesis_scores: list[float] = []
        contextual_precision_scores: list[float] = []
        contextual_recall_scores: list[float] = []
        faithfulness_scores: list[float] = []
        answer_relevancy_scores: list[float] = []
        retrieval_precision_scores: list[float] = []
        semantic_quality_scores: list[float] = []
        recovery_scores: list[float] = []
        refinement_gains: list[float] = []
        evidence_consistency_scores: list[float] = []
        contradiction_handling_scores: list[float] = []
        iterative_grounding_scores: list[float] = []
        plan_completion_scores: list[float] = []
        reasoning_depth_scores: list[float] = []
        sub_query_effectiveness_scores: list[float] = []
        for probe in probes:
            answer = self.pipeline.run(
                probe.query,
                hybrid=True,
                rerank=True,
                expand_query=True,
                context_window=True,
                multi_hop=False,
                structured_answer=True,
                query_understanding=True,
                compression=True,
                answerability_filter=True,
                section_routing=True,
                agentic=True,
                reflection=True,
                iterative_retrieval=True,
                evidence_graph=True,
                refine_answer=True,
                detect_contradictions=True,
                observability=False,
                save_session=False,
            )
            grounding_scores.append(answer.grounding_report.grounding_score if answer.grounding_report else 0.0)
            citation_scores.append(min(len(answer.citations) / max(len(answer.evidence_chunks), 1), 1.0))
            relevance_scores.append(1.0 if any(item.chunk_id in probe.relevant_chunk_ids for item in answer.evidence_chunks) else 0.0)
            completeness_scores.append(answer.answer_quality_report.semantic_completeness if answer.answer_quality_report else 0.0)
            hallucination_scores.append(answer.hallucination_score)
            latency_scores.append(float(answer.generation_metadata.latency_ms if answer.generation_metadata else 0.0))
            synthesis_scores.append(answer.answer_quality_report.synthesis_quality if answer.answer_quality_report else 0.0)
            contextual_precision_scores.append(min(len([item for item in answer.evidence_chunks if item.chunk_id in probe.relevant_chunk_ids]) / max(len(answer.evidence_chunks), 1), 1.0))
            contextual_recall_scores.append(min(len([item for item in answer.evidence_chunks if item.chunk_id in probe.relevant_chunk_ids]) / max(len(probe.relevant_chunk_ids), 1), 1.0))
            faithfulness_scores.append(answer.grounding_report.grounding_score if answer.grounding_report else 0.0)
            answer_relevancy_scores.append(answer.answer_quality_report.retrieval_alignment if answer.answer_quality_report else 0.0)
            retrieval_precision_scores.append(min(len([item for item in answer.evidence_chunks if item.chunk_id in probe.relevant_chunk_ids]) / max(len(answer.evidence_chunks), 1), 1.0))
            semantic_quality_scores.append(answer.answer_quality_report.overall_answer_quality_score if answer.answer_quality_report else 0.0)
            retrieval_loop = answer.retrieval_metadata.get("retrieval_loop_report", {})
            recovery_scores.append(1.0 if retrieval_loop and float(retrieval_loop.get("coverage_score", 0.0)) >= 0.6 else 0.0)
            refinement_gains.append(float(answer.retrieval_metadata.get("refinement_gain", 0.0)))
            contradiction_report = answer.retrieval_metadata.get("contradiction_report", {})
            evidence_consistency_scores.append(max(0.0, 1.0 - float(contradiction_report.get("contradiction_score", 0.0))))
            contradiction_handling_scores.append(1.0 if contradiction_report else 0.0)
            iterative_grounding_scores.append(answer.grounding_report.grounding_score if answer.grounding_report else 0.0)
            plan = answer.retrieval_metadata.get("agentic_plan", {})
            subtasks = plan.get("subtasks", []) if isinstance(plan, dict) else []
            plan_completion_scores.append(1.0 if subtasks else 0.0)
            reasoning_depth_scores.append(float(answer.retrieval_metadata.get("reasoning_steps_used", 0)))
            sub_query_effectiveness_scores.append(min(len(answer.evidence_chunks) / max(len(subtasks), 1), 1.0) if subtasks else 0.0)
        report = RAGEvaluationReport(
            evaluation_id=f"rag-{utc_now().strftime('%Y%m%d%H%M%S')}",
            probe_count=len(probes),
            answer_grounding=round(sum(grounding_scores) / max(len(grounding_scores), 1), 6),
            citation_correctness=round(sum(citation_scores) / max(len(citation_scores), 1), 6),
            retrieval_relevance=round(sum(relevance_scores) / max(len(relevance_scores), 1), 6),
            answer_completeness=round(sum(completeness_scores) / max(len(completeness_scores), 1), 6),
            hallucination_rate=round(sum(hallucination_scores) / max(len(hallucination_scores), 1), 6),
            latency_ms_mean=round(sum(latency_scores) / max(len(latency_scores), 1), 6),
            synthesis_quality=round(sum(synthesis_scores) / max(len(synthesis_scores), 1), 6),
            contextual_precision=round(sum(contextual_precision_scores) / max(len(contextual_precision_scores), 1), 6),
            contextual_recall=round(sum(contextual_recall_scores) / max(len(contextual_recall_scores), 1), 6),
            faithfulness=round(sum(faithfulness_scores) / max(len(faithfulness_scores), 1), 6),
            answer_relevancy=round(sum(answer_relevancy_scores) / max(len(answer_relevancy_scores), 1), 6),
            retrieval_precision=round(sum(retrieval_precision_scores) / max(len(retrieval_precision_scores), 1), 6),
            semantic_answer_quality=round(sum(semantic_quality_scores) / max(len(semantic_quality_scores), 1), 6),
            retrieval_recovery_rate=round(sum(recovery_scores) / max(len(recovery_scores), 1), 6),
            refinement_gain=round(sum(refinement_gains) / max(len(refinement_gains), 1), 6),
            evidence_consistency=round(sum(evidence_consistency_scores) / max(len(evidence_consistency_scores), 1), 6),
            contradiction_handling=round(sum(contradiction_handling_scores) / max(len(contradiction_handling_scores), 1), 6),
            iterative_grounding_improvement=round(sum(iterative_grounding_scores) / max(len(iterative_grounding_scores), 1), 6),
            plan_completion_quality=round(sum(plan_completion_scores) / max(len(plan_completion_scores), 1), 6),
            reasoning_depth=round(sum(reasoning_depth_scores) / max(len(reasoning_depth_scores), 1), 6),
            sub_query_effectiveness=round(sum(sub_query_effectiveness_scores) / max(len(sub_query_effectiveness_scores), 1), 6),
            metadata={"probe_count": len(probes), "ragas_style_metrics": True, "agentic_metrics": True},
        )
        write_json(self.config.rag.rag_evaluation_dir / f"{report.evaluation_id}.json", report.model_dump(mode="json"))
        return report
