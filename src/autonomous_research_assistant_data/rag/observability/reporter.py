"""Persist retrieval and answer observability metrics."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import GroundingReport, RAGObservabilityReport
from autonomous_research_assistant_data.storage.file_store import append_jsonl, write_json


class RAGObservabilityReporter:
    """Write observability metrics for RAG runs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        query: str,
        retrieval_latency_ms: float,
        reranker_lift: float,
        chunk_utilization: float,
        context_waste_ratio: float,
        prompt_efficiency: float,
        grounding: GroundingReport,
        reasoning_depth: int = 0,
        retrieval_retries: int = 0,
        refinement_gain: float = 0.0,
        evidence_graph_nodes: int = 0,
        evidence_graph_edges: int = 0,
        extra_metadata: dict[str, object] | None = None,
    ) -> RAGObservabilityReport:
        return RAGObservabilityReport(
            query=query,
            retrieval_drift=round(max(0.0, 1.0 - grounding.grounding_score), 6),
            reranker_lift=round(reranker_lift, 6),
            chunk_utilization=round(chunk_utilization, 6),
            context_waste_ratio=round(context_waste_ratio, 6),
            hallucination_hotspot_score=round(grounding.hallucination_probability, 6),
            unsupported_claim_frequency=round(grounding.unsupported_claim_ratio, 6),
            retrieval_latency_ms=round(retrieval_latency_ms, 6),
            prompt_efficiency=round(prompt_efficiency, 6),
            reasoning_depth=reasoning_depth,
            retrieval_retries=retrieval_retries,
            refinement_gain=round(refinement_gain, 6),
            evidence_graph_nodes=evidence_graph_nodes,
            evidence_graph_edges=evidence_graph_edges,
            metadata={"generated_at": utc_now(), **dict(extra_metadata or {})},
        )

    def write(self, report: RAGObservabilityReport) -> None:
        payload = report.model_dump(mode="json")
        write_json(self.config.rag.rag_observability_dir / "latest_observability.json", payload)
        append_jsonl(self.config.rag.rag_observability_dir / "observability_log.jsonl", payload)
