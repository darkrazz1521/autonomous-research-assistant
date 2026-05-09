"""Fusion methods for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Iterable

from autonomous_research_assistant_data.config import AppConfig


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def reciprocal_rank_fusion(config: AppConfig, dense: list[tuple[str, float]], sparse: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    k = config.retrieval.fusion.rrf_k
    for rank, (chunk_id, score) in enumerate(dense, start=1):
        merged.setdefault(chunk_id, {"dense": 0.0, "sparse": 0.0, "fused": 0.0})
        merged[chunk_id]["dense"] = score
        merged[chunk_id]["fused"] += 1.0 / (k + rank)
    for rank, (chunk_id, score) in enumerate(sparse, start=1):
        merged.setdefault(chunk_id, {"dense": 0.0, "sparse": 0.0, "fused": 0.0})
        merged[chunk_id]["sparse"] = score
        merged[chunk_id]["fused"] += 1.0 / (k + rank)
    return merged


def weighted_fusion(config: AppConfig, dense: list[tuple[str, float]], sparse: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
    dense_map = dict(dense)
    sparse_map = dict(sparse)
    dense_norm = _normalize_scores(dense_map) if config.retrieval.fusion.normalize_dense_sparse else dense_map
    sparse_norm = _normalize_scores(sparse_map) if config.retrieval.fusion.normalize_dense_sparse else sparse_map
    merged: dict[str, dict[str, float]] = {}
    for chunk_id in set(dense_map).union(sparse_map):
        dense_score = dense_map.get(chunk_id, 0.0)
        sparse_score = sparse_map.get(chunk_id, 0.0)
        fused = (dense_norm.get(chunk_id, 0.0) * config.retrieval.fusion.dense_weight) + (
            sparse_norm.get(chunk_id, 0.0) * config.retrieval.fusion.sparse_weight
        )
        merged[chunk_id] = {"dense": dense_score, "sparse": sparse_score, "fused": fused}
    return merged


def fuse_results(config: AppConfig, dense: list[tuple[str, float]], sparse: list[tuple[str, float]], *, method: str | None = None) -> dict[str, dict[str, float]]:
    resolved = (method or config.retrieval.fusion.method or config.retrieval.search.hybrid_fusion).lower()
    if resolved == "weighted":
        return weighted_fusion(config, dense, sparse)
    return reciprocal_rank_fusion(config, dense, sparse)


def rerank_aware_final_score(config: AppConfig, *, fused_score: float, section_weight: float, rerank_score: float | None = None, citation_boost: float = 0.0) -> float:
    rerank_component = 0.0 if rerank_score is None else rerank_score * config.retrieval.fusion.rerank_weight
    return round((fused_score + rerank_component + citation_boost) * section_weight, 6)
