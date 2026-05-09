"""Hybrid dense + sparse retrieval."""

from __future__ import annotations

import re
from time import perf_counter

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import EmbeddingRecord, RetrievalResult, RetrievalTrace
from autonomous_research_assistant_data.retrieval.embedding.service import EmbeddingService
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

    def _result_from_record(self, record: EmbeddingRecord, *, score: float, dense_score: float | None, sparse_score: float | None) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=record.chunk_id,
            paper_id=record.paper_id,
            arxiv_id=record.arxiv_id,
            score=score,
            dense_score=dense_score,
            sparse_score=sparse_score,
            chunk_text=record.chunk_text,
            section_name=str(record.metadata.get("section_name", "Unknown")),
            canonical_section_label=record.metadata.get("canonical_section_label"),
            citations=[span.get("text", "") for span in record.metadata.get("citation_spans", []) if isinstance(span, dict)],
            citation_entities=list(record.metadata.get("citation_entities", [])),
            neighboring_chunk_ids=[value for value in [record.metadata.get("previous_chunk_id"), record.metadata.get("next_chunk_id")] if value],
            metadata=record.metadata,
        )

    def _rrf(self, dense: list[tuple[str, float]], sparse: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
        merged: dict[str, dict[str, float]] = {}
        k = self.config.retrieval.search.rrf_k
        for rank, (chunk_id, score) in enumerate(dense, start=1):
            merged.setdefault(chunk_id, {"dense": 0.0, "sparse": 0.0, "score": 0.0})
            merged[chunk_id]["dense"] = score
            merged[chunk_id]["score"] += 1.0 / (k + rank)
        for rank, (chunk_id, score) in enumerate(sparse, start=1):
            merged.setdefault(chunk_id, {"dense": 0.0, "sparse": 0.0, "score": 0.0})
            merged[chunk_id]["sparse"] = score
            merged[chunk_id]["score"] += 1.0 / (k + rank)
        return merged

    def _weighted_fusion(self, dense: list[tuple[str, float]], sparse: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
        merged: dict[str, dict[str, float]] = {}
        dense_map = dict(dense)
        sparse_map = dict(sparse)
        max_dense = max([abs(score) for score in dense_map.values()] or [1.0])
        max_sparse = max([abs(score) for score in sparse_map.values()] or [1.0])
        for chunk_id in set(dense_map).union(sparse_map):
            dense_score = dense_map.get(chunk_id, 0.0) / max_dense
            sparse_score = sparse_map.get(chunk_id, 0.0) / max_sparse
            merged[chunk_id] = {
                "dense": dense_map.get(chunk_id, 0.0),
                "sparse": sparse_map.get(chunk_id, 0.0),
                "score": dense_score * self.config.retrieval.search.dense_weight + sparse_score * self.config.retrieval.search.sparse_weight,
            }
        return merged

    def search(self, query: str, *, top_k: int, namespace: str, mode: str = "hybrid", section_filter: str | None = None) -> RetrievalTrace:
        started = perf_counter()
        dense_started = perf_counter()
        query_vector = self.embedder.encode([query], query=True, batch_size=1)[0].tolist()
        dense_hits = self.store.search(query_vector, top_k=self.config.retrieval.search.dense_top_k, namespace=namespace)
        dense_latency = (perf_counter() - dense_started) * 1000

        sparse_started = perf_counter()
        sparse_hits = self.bm25.score(query, top_k=self.config.retrieval.search.sparse_top_k)
        sparse_latency = (perf_counter() - sparse_started) * 1000

        if mode == "dense":
            merged = {chunk_id: {"dense": score, "sparse": 0.0, "score": score} for chunk_id, score in dense_hits}
        elif mode == "sparse":
            merged = {chunk_id: {"dense": 0.0, "sparse": score, "score": score} for chunk_id, score in sparse_hits}
        elif self.config.retrieval.search.hybrid_fusion == "weighted":
            merged = self._weighted_fusion(dense_hits, sparse_hits)
        else:
            merged = self._rrf(dense_hits, sparse_hits)

        results: list[RetrievalResult] = []
        for chunk_id, scores in merged.items():
            record = self.store.records.get(chunk_id)
            if record is None:
                continue
            if section_filter and str(record.metadata.get("canonical_section_label")) != section_filter:
                continue
            citation_boost = self._citation_boost(query, record)
            section_boost = self._section_boost(query, record) if self.config.retrieval.search.section_aware_boost else 0.0
            result = self._result_from_record(
                record,
                score=float(scores["score"] + citation_boost + section_boost),
                dense_score=float(scores.get("dense", 0.0)),
                sparse_score=float(scores.get("sparse", 0.0)),
            )
            result.citation_boost = citation_boost
            result.section_boost = section_boost
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
            metadata={"section_filter": section_filter},
        )

