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
        for probe in probes:
            answer = self.pipeline.run(
                probe.query,
                hybrid=True,
                rerank=True,
                expand_query=True,
                context_window=True,
                multi_hop=False,
                save_session=False,
            )
            grounding_scores.append(answer.grounding_report.grounding_score if answer.grounding_report else 0.0)
            citation_scores.append(min(len(answer.citations) / max(len(answer.evidence_chunks), 1), 1.0))
            relevance_scores.append(1.0 if any(item.chunk_id in probe.relevant_chunk_ids for item in answer.evidence_chunks) else 0.0)
            completeness_scores.append(answer.answer_quality_report.semantic_completeness if answer.answer_quality_report else 0.0)
            hallucination_scores.append(answer.hallucination_score)
            latency_scores.append(float(answer.generation_metadata.latency_ms if answer.generation_metadata else 0.0))
            synthesis_scores.append(answer.answer_quality_report.synthesis_quality if answer.answer_quality_report else 0.0)
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
            metadata={"probe_count": len(probes)},
        )
        write_json(self.config.rag.rag_evaluation_dir / f"{report.evaluation_id}.json", report.model_dump(mode="json"))
        return report
