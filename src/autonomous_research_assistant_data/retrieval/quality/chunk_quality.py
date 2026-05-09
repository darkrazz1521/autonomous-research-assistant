"""Chunk-level retrieval quality analysis with cached analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import ChunkRecord
from autonomous_research_assistant_data.retrieval.quality.noise_classifier import (
    alphabetic_ratio,
    citation_density,
    classify_noise,
    duplicate_line_ratio,
    equation_density,
    numeric_ratio,
)
from autonomous_research_assistant_data.retrieval.quality.semantic_density import (
    average_sentence_length,
    language_entropy,
    semantic_density_score,
)
from autonomous_research_assistant_data.retrieval.quality.table_detector import (
    detect_benchmark_probability,
    detect_table_probability,
    malformed_structure_score,
)
from autonomous_research_assistant_data.storage.file_store import ensure_directory, read_json, write_json


class ChunkQualityAnalyzer:
    """Compute, cache, and report retrieval quality for chunk records."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cache_dir = ensure_directory(config.retrieval.retrieval_analytics_dir / "chunk_quality")

    def _cache_path(self, chunk: ChunkRecord, source_path: Path) -> Path:
        relative = source_path.relative_to(self.config.pdf_processing.chunks_dir)
        return self.cache_dir / relative.parent / f"{chunk.chunk_id}.json"

    def analyse(self, chunk: ChunkRecord, *, source_path: Path) -> dict[str, Any]:
        cache_path = self._cache_path(chunk, source_path)
        if self.config.retrieval.quality.cache_enabled:
            cached = read_json(cache_path, default={})
            if cached:
                return cached

        text = chunk.chunk_text
        semantic_density = semantic_density_score(text)
        table_probability = detect_table_probability(text)
        benchmark_probability = detect_benchmark_probability(text)
        malformed = malformed_structure_score(text)
        numeric_share = numeric_ratio(text)
        alphabetic_share = alphabetic_ratio(text)
        duplicate_ratio = duplicate_line_ratio(text)
        citation_share = citation_density(text)
        equation_share = equation_density(text)
        entropy = language_entropy(text)
        avg_sentence_len = average_sentence_length(text)
        noise_score, labels = classify_noise(
            semantic_density=semantic_density,
            table_probability=table_probability,
            benchmark_probability=benchmark_probability,
            duplicate_ratio=duplicate_ratio,
            numeric_share=numeric_share,
            malformed_structure=malformed,
        )
        quality_score = max(
            0.0,
            min(
                1.0,
                (semantic_density * 0.30)
                + (alphabetic_share * 0.12)
                + ((1 - table_probability) * 0.15)
                + ((1 - benchmark_probability) * 0.10)
                + ((1 - duplicate_ratio) * 0.08)
                + ((1 - malformed) * 0.10)
                + min(entropy, 1.0) * 0.05
                + min(avg_sentence_len / 24.0, 1.0) * 0.05
                + ((1 - min(numeric_share / 0.25, 1.0)) * 0.05),
            ),
        )
        excluded = (
            semantic_density < self.config.retrieval.quality.min_semantic_density
            or table_probability > self.config.retrieval.quality.max_table_probability
            or noise_score > self.config.retrieval.quality.max_noise_score
            or quality_score < self.config.retrieval.quality.min_quality_score
        )
        payload = {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "source_chunk_path": str(source_path),
            "numeric_ratio": numeric_share,
            "alphabetic_ratio": alphabetic_share,
            "semantic_density_score": semantic_density,
            "language_entropy": entropy,
            "table_probability": table_probability,
            "benchmark_probability": benchmark_probability,
            "duplicate_line_ratio": duplicate_ratio,
            "citation_density": citation_share,
            "equation_density": equation_share,
            "average_sentence_length": avg_sentence_len,
            "malformed_structure_score": malformed,
            "retrieval_quality_score": round(quality_score, 6),
            "retrieval_noise_score": noise_score,
            "retrieval_excluded": excluded,
            "noise_classifications": sorted(set(labels)),
        }
        write_json(cache_path, payload)
        return payload
