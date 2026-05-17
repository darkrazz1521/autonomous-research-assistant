"""MMR-style compression, pruning, and deduplication."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult
from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class ContextCompressor:
    """Reduce noisy context while preserving answer-bearing evidence."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _cache_path(self, cache_key: str) -> Path:
        return self.config.rag.rag_cache_dir / "compression" / f"{cache_key}.json"

    def _sentence_split(self, text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    def _similarity(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]+", right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens.intersection(right_tokens)) / max(len(left_tokens.union(right_tokens)), 1)

    def compress(
        self,
        query: str,
        results: list[RetrievalResult],
        understanding: QueryUnderstandingResult,
        *,
        use_mmr: bool = False,
        enable_compression: bool = False,
    ) -> tuple[list[RetrievalResult], dict[str, float]]:
        if not enable_compression:
            return results, {"context_waste_ratio": 0.0, "chunk_utilization": 1.0}
        cache_key = hashlib.sha256(f"{query}:{understanding.query_type}:{','.join(item.chunk_id for item in results[:12])}:{use_mmr}".encode("utf-8")).hexdigest()
        if self.config.rag.context_processing.compression_cache_enabled:
            cached = read_json(self._cache_path(cache_key), default={})
            if cached:
                keep_ids = set(cached.get("kept_chunk_ids", []))
                restored = [item for item in results if item.chunk_id in keep_ids]
                return restored, {
                    "context_waste_ratio": float(cached.get("context_waste_ratio", 0.0)),
                    "chunk_utilization": float(cached.get("chunk_utilization", 1.0)),
                }
        kept: list[RetrievalResult] = []
        for result in results:
            candidate = result.model_copy(deep=True)
            text = candidate.merged_context or candidate.chunk_text
            candidate.merged_context = " ".join(self._sentence_split(text)[: self.config.rag.context_processing.max_sentences_per_chunk])
            if not kept:
                kept.append(candidate)
                continue
            similarity = max(self._similarity(candidate.merged_context, prior.merged_context or prior.chunk_text) for prior in kept)
            if similarity > 0.72:
                continue
            if use_mmr:
                mmr_score = (self.config.rag.context_processing.mmr_lambda * candidate.score) - ((1 - self.config.rag.context_processing.mmr_lambda) * similarity)
                candidate.final_score_breakdown["mmr_score"] = round(mmr_score, 6)
                if mmr_score < 0:
                    continue
            kept.append(candidate)
        target_count = max(1, int(len(results) * (1 - self.config.rag.context_processing.target_reduction_ratio)))
        compressed = kept[:target_count]
        metrics = {
            "context_waste_ratio": round(max(len(results) - len(compressed), 0) / max(len(results), 1), 6),
            "chunk_utilization": round(len(compressed) / max(len(results), 1), 6),
        }
        write_json(self._cache_path(cache_key), {"kept_chunk_ids": [item.chunk_id for item in compressed], **metrics})
        return compressed, metrics
