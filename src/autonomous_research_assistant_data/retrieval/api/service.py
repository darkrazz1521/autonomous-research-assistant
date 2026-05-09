"""High-level retrieval API."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import RetrievalTrace
from autonomous_research_assistant_data.retrieval.analytics.reporter import RetrievalAnalyticsReporter
from autonomous_research_assistant_data.retrieval.rerank.service import RerankerService
from autonomous_research_assistant_data.retrieval.search.hybrid import HybridRetrievalEngine
from autonomous_research_assistant_data.retrieval.vectorstores.registry import get_vector_store


class RetrievalApi:
    """Search retrieval-grade scientific chunks."""

    def __init__(self, config: AppConfig, *, model_name: str | None = None, backend: str | None = None) -> None:
        self.config = config
        self.model_name = model_name or config.retrieval.embedding.default_model
        self.namespace = config.retrieval.vector_db.namespace
        self.store = get_vector_store(config, self.model_name, backend)
        self.store.load(namespace=self.namespace)
        self.hybrid = HybridRetrievalEngine(config, self.model_name, self.store)
        self.reranker = RerankerService(config)
        self.analytics = RetrievalAnalyticsReporter(config.retrieval.retrieval_analytics_dir)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        mode: str = "hybrid",
        rerank: bool = False,
        citation_aware: bool = True,
        section_filter: str | None = None,
    ) -> RetrievalTrace:
        trace = self.hybrid.search(query, top_k=top_k, namespace=self.namespace, mode=mode, section_filter=section_filter)
        if rerank and self.config.retrieval.reranker.enabled:
            reranked = self.reranker.rerank(query, trace.results[: self.config.retrieval.reranker.top_k_depth])
            trace.results = reranked + trace.results[self.config.retrieval.reranker.top_k_depth :]
            trace.rerank_latency_ms = 0.0
        if not citation_aware:
            for item in trace.results:
                item.score -= item.citation_boost
                item.citation_boost = 0.0
        self.analytics.append_trace(trace)
        return trace

