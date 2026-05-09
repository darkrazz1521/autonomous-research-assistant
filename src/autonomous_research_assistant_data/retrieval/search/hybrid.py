"""Hybrid dense + sparse retrieval."""

from __future__ import annotations

import re
from time import perf_counter

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import EmbeddingRecord, RetrievalResult, RetrievalTrace
from autonomous_research_assistant_data.retrieval.context.windowing import ContextWindowBuilder
from autonomous_research_assistant_data.retrieval.embedding.service import EmbeddingService
from autonomous_research_assistant_data.retrieval.query_expansion.expander import QueryExpander
from autonomous_research_assistant_data.retrieval.ranking.fusion import fuse_results, rerank_aware_final_score
from autonomous_research_assistant_data.retrieval.ranking.section_weights import resolve_section_weight
from autonomous_research_assistant_data.retrieval.search.bm25 import BM25Index
from autonomous_research_assistant_data.retrieval.vectorstores.faiss_store import FaissVectorStore


class HybridRetrievalEngine:
    """Search scientific chunks using dense, sparse, or fused retrieval."""

    def __init__(self, config: AppConfig, model_name: str, store: FaissVectorStore) -> None:
        self.config = config
        self.model_name = model_name
        self.store = store
        self.embedder = EmbeddingService(
            model_name,
            normalize=config.retrieval.embedding.normalize_embeddings,
            max_length=config.retrieval.embedding.max_length,
            device=config.retrieval.embedding.device,
            fallback_dim=config.retrieval.embedding.deterministic_fallback_dim,
        )
        documents = {chunk_id: record.chunk_text for chunk_id, record in store.records.items()}
        self.bm25 = BM25Index(documents)
        self.query_expander = QueryExpander(config)
        self.window_builder = ContextWindowBuilder(config, store.records)

    def _citation_query_entities(self, query: str) -> set[str]:
        return set(re.findall(r"[A-Z][A-Za-z]+(?:\s+et al\.)?", query))

    def _citation_boost(self, query: str, record: EmbeddingRecord) -> float:
        boost = 0.0
        entities = self._citation_query_entities(query)
        record_entities = {str(entity) for entity in record.metadata.get("citation_entities", [])}
        if entities.intersection(record_entities):
            boost += 0.08
        if "citation" in query.lower() and record.metadata.get("citation_density", 0.0) > 0:
            boost += 0.04
        return boost

    def _section_boost(self, query: str, record: EmbeddingRecord) -> float:
        label = str(record.metadata.get("canonical_section_label") or "")
        lowered = query.lower()
        if "method" in lowered and label in {"methodology", "preliminaries"}:
            return 0.05
        if "result" in lowered and label in {"results", "discussion", "experiments"}:
            return 0.05
        if "conclusion" in lowered and label == "conclusion":
            return 0.05
        return 0.0

    def _result_from_record(
        self,
        record: EmbeddingRecord,
        *,
        score: float,
        dense_score: float | None,
        sparse_score: float | None,
        section_weight: float,
        fused_score: float,
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=record.chunk_id,
            paper_id=record.paper_id,
            arxiv_id=record.arxiv_id,
            score=score,
            dense_score=dense_score,
            sparse_score=sparse_score,
            raw_vector_score=dense_score,
            raw_sparse_score=sparse_score,
            section_weight=section_weight,
            final_retrieval_score=score,
            final_score_breakdown={"fused_score": fused_score},
            chunk_text=record.chunk_text,
            section_name=str(record.metadata.get("section_name", "Unknown")),
            canonical_section_label=record.metadata.get("canonical_section_label"),
            citations=[span.get("text", "") for span in record.metadata.get("citation_spans", []) if isinstance(span, dict)],
            citation_entities=list(record.metadata.get("citation_entities", [])),
            neighboring_chunk_ids=[value for value in [record.metadata.get("previous_chunk_id"), record.metadata.get("next_chunk_id")] if value],
            metadata=record.metadata,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        namespace: str,
        mode: str = "hybrid",
        section_filter: str | None = None,
        fusion_method: str | None = None,
        expand_query: bool = False,
        section_weighting_enabled: bool = True,
        context_window: bool = False,
        window_radius: int | None = None,
    ) -> RetrievalTrace:
        started = perf_counter()
        expansion_report = self.query_expander.expand(query, enabled=expand_query)
        expanded_query = str(expansion_report["rewritten_query"])
        dense_started = perf_counter()
        query_vector = self.embedder.encode([expanded_query], query=True, batch_size=1)[0].tolist()
        dense_hits = self.store.search(query_vector, top_k=self.config.retrieval.search.dense_top_k, namespace=namespace)
        dense_latency = (perf_counter() - dense_started) * 1000

        sparse_started = perf_counter()
        sparse_hits = self.bm25.score(expanded_query, top_k=self.config.retrieval.search.sparse_top_k)
        sparse_latency = (perf_counter() - sparse_started) * 1000

        if mode == "dense":
            merged = {chunk_id: {"dense": score, "sparse": 0.0, "fused": score} for chunk_id, score in dense_hits}
        elif mode == "sparse":
            merged = {chunk_id: {"dense": 0.0, "sparse": score, "fused": score} for chunk_id, score in sparse_hits}
        else:
            merged = fuse_results(self.config, dense_hits, sparse_hits, method=fusion_method)

        results: list[RetrievalResult] = []
        for chunk_id, scores in merged.items():
            record = self.store.records.get(chunk_id)
            if record is None:
                continue
            if section_filter and str(record.metadata.get("canonical_section_label")) != section_filter:
                continue
            citation_boost = self._citation_boost(expanded_query, record)
            section_boost = self._section_boost(expanded_query, record) if self.config.retrieval.search.section_aware_boost else 0.0
            section_weight = resolve_section_weight(
                self.config,
                str(record.metadata.get("canonical_section_label") or ""),
                enabled=section_weighting_enabled,
            )
            fused_score = float(scores.get("fused", 0.0)) + section_boost
            final_score = rerank_aware_final_score(
                self.config,
                fused_score=fused_score,
                section_weight=section_weight,
                citation_boost=citation_boost,
            )
            result = self._result_from_record(
                record,
                score=final_score,
                dense_score=float(scores.get("dense", 0.0)),
                sparse_score=float(scores.get("sparse", 0.0)),
                section_weight=section_weight,
                fused_score=fused_score,
            )
            result.citation_boost = citation_boost
            result.section_boost = section_boost
            result.final_score_breakdown.update(
                {
                    "raw_vector_score": result.raw_vector_score,
                    "raw_sparse_score": result.raw_sparse_score,
                    "citation_boost": citation_boost,
                    "section_boost": section_boost,
                    "section_weight": section_weight,
                    "fusion_method": fusion_method or self.config.retrieval.fusion.method,
                }
            )
            if context_window:
                result = self.window_builder.enrich(result, enabled=True, radius=window_radius)
            else:
                result = self.window_builder.enrich(result, enabled=False)
            results.append(result)

        results.sort(key=lambda item: item.score, reverse=True)
        results = results[:top_k]
        for rank, item in enumerate(results, start=1):
            item.rank = rank

        total_latency = (perf_counter() - started) * 1000
        return RetrievalTrace(
            query=query,
            mode=mode,
            namespace=namespace,
            top_k=top_k,
            latency_ms=round(total_latency, 3),
            dense_latency_ms=round(dense_latency, 3),
            sparse_latency_ms=round(sparse_latency, 3),
            results=results,
            metadata={
                "section_filter": section_filter,
                "query_expansion_report": expansion_report,
                "context_window_enabled": context_window,
                "window_radius": window_radius,
                "fusion_method": fusion_method or self.config.retrieval.fusion.method,
            },
        )
