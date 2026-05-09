"""Cross-encoder reranking with heuristic fallback."""

from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import RetrievalResult
from autonomous_research_assistant_data.retrieval.common import slugify_model_name, stable_hash_text
from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class RerankerService:
    """Rerank retrieval candidates while preserving explainability."""

    def __init__(self, config: AppConfig, model_name: str | None = None) -> None:
        self.config = config
        self.model_name = model_name or config.retrieval.reranker.default_model
        self.model_slug = slugify_model_name(self.model_name)
        self._model = None
        self._backend = "heuristic"
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            sentence_transformers = import_module("sentence_transformers")
            CrossEncoder = getattr(sentence_transformers, "CrossEncoder")
            self._model = CrossEncoder(self.model_name)
            self._backend = "cross-encoder"
        except Exception:
            self._model = None
            self._backend = "heuristic"

    def _cache_path(self, query: str, chunk_ids: list[str]) -> Path:
        digest = stable_hash_text(f"{self.model_slug}:{query}:{'|'.join(chunk_ids)}")
        return self.config.retrieval.rerank_cache_dir / self.model_slug / f"{digest}.json"

    def _heuristic_score(self, query: str, result: RetrievalResult) -> float:
        query_terms = set(re.findall(r"[a-z0-9][a-z0-9_\-\.]+", query.lower()))
        text_terms = set(re.findall(r"[a-z0-9][a-z0-9_\-\.]+", result.chunk_text.lower()))
        overlap = len(query_terms.intersection(text_terms)) / max(len(query_terms), 1)
        score = overlap
        if result.citation_entities:
            score += self.config.retrieval.reranker.citation_sensitivity_boost
        if result.canonical_section_label in {"introduction", "methodology", "results", "discussion", "conclusion"}:
            score += self.config.retrieval.reranker.section_priority_boost
        return round(score, 6)

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []
        chunk_ids = [item.chunk_id for item in results]
        cache_path = self._cache_path(query, chunk_ids)
        cached = read_json(cache_path, default={})
        if cached:
            scores = cached.get("scores", {})
        elif self._model is not None:
            pairs = [[query, item.chunk_text] for item in results]
            raw_scores = self._model.predict(pairs)
            scores = {item.chunk_id: float(score) for item, score in zip(results, raw_scores, strict=True)}
            write_json(cache_path, {"backend": self._backend, "scores": scores})
        else:
            scores = {item.chunk_id: self._heuristic_score(query, item) for item in results}
            write_json(cache_path, {"backend": self._backend, "scores": scores})

        reranked = []
        for item in results:
            updated = item.model_copy(deep=True)
            updated.rerank_score = float(scores.get(item.chunk_id, item.score))
            updated.score = updated.rerank_score
            reranked.append(updated)
        reranked.sort(key=lambda item: item.score, reverse=True)
        for rank, item in enumerate(reranked, start=1):
            item.rank = rank
        return reranked

