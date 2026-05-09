"""Embedding generation pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import EmbeddingRecord
from autonomous_research_assistant_data.retrieval.analytics.reporter import RetrievalAnalyticsReporter
from autonomous_research_assistant_data.retrieval.common import load_chunk_records, slugify_model_name, stable_hash_text
from autonomous_research_assistant_data.retrieval.embedding.service import EmbeddingService
from autonomous_research_assistant_data.storage.file_store import read_json, write_json
from autonomous_research_assistant_data.storage.manifest import ManifestStore


class EmbeddingPipeline:
    """Generate retrieval embeddings from processed chunks."""

    source_name = "embedding_pipeline"

    def __init__(self, config: AppConfig, *, model_name: str | None = None, batch_size: int | None = None) -> None:
        self.config = config
        self.logger = get_logger(f"retrieval.{self.source_name}")
        self.model_name = model_name or config.retrieval.embedding.default_model
        self.batch_size = batch_size or config.retrieval.embedding.batch_size
        self.model_slug = slugify_model_name(self.model_name)
        self.manifest = ManifestStore(config.retrieval.manifests_dir / "embeddings_manifest.json")
        self.analytics = RetrievalAnalyticsReporter(config.retrieval.retrieval_analytics_dir)
        self.embedder = EmbeddingService(
            self.model_name,
            normalize=config.retrieval.embedding.normalize_embeddings,
            max_length=config.retrieval.embedding.max_length,
            device=config.retrieval.embedding.device,
            fallback_dim=config.retrieval.embedding.deterministic_fallback_dim,
        )

    def _output_path(self, chunk_source: Path, paper_id: str) -> Path:
        relative = chunk_source.relative_to(self.config.pdf_processing.chunks_dir)
        parts = relative.parts[:-1]
        return self.config.retrieval.embeddings_dir / self.model_slug / Path(*parts) / f"{paper_id}.json"

    def _eligible(self, chunk) -> bool:
        cfg = self.config.retrieval.embedding
        if chunk.retrieval_quality_score < cfg.min_retrieval_quality_score:
            return False
        if chunk.flagged_for_review and not cfg.include_flagged_chunks:
            return False
        return True

    def generate(self, *, force_rebuild: bool = False, namespace: str | None = None) -> dict[str, Any]:
        namespace_value = namespace or self.config.retrieval.vector_db.namespace
        chunk_records = [(path, chunk) for path, chunk in load_chunk_records(self.config.pdf_processing.chunks_dir) if self._eligible(chunk)]
        grouped: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
        for path, chunk in chunk_records:
            grouped[chunk.paper_id].append((path, chunk))

        written_records = 0
        skipped_records = 0
        output_files: list[str] = []

        for paper_id in sorted(grouped):
            paper_chunks = grouped[paper_id]
            payload_records: list[EmbeddingRecord] = []
            for batch_start in range(0, len(paper_chunks), self.batch_size):
                batch = paper_chunks[batch_start : batch_start + self.batch_size]
                texts = [item[1].chunk_text for item in batch]
                vectors = self.embedder.encode(texts, batch_size=self.batch_size)
                for (source_path, chunk), vector in zip(batch, vectors, strict=True):
                    entry_id = f"{self.model_slug}:{namespace_value}:{chunk.chunk_id}"
                    if not force_rebuild and self.manifest.exists(entry_id):
                        skipped_records += 1
                        continue
                    vector_list = [round(float(value), 8) for value in vector.tolist()]
                    record = EmbeddingRecord(
                        embedding_id=stable_hash_text(entry_id),
                        chunk_id=chunk.chunk_id,
                        paper_id=chunk.paper_id,
                        arxiv_id=chunk.arxiv_id,
                        model_name=self.model_name,
                        vector_dim=len(vector_list),
                        vector_norm=round(float(sum(value * value for value in vector_list) ** 0.5), 6),
                        normalized=self.config.retrieval.embedding.normalize_embeddings,
                        namespace=namespace_value,
                        chunk_text=chunk.chunk_text,
                        metadata={
                            "section_name": chunk.section_name,
                            "canonical_section_label": chunk.extra.get("canonical_section_label"),
                            "page_range": list(chunk.page_range),
                            "source_pdf": str(chunk.source_pdf),
                            "semantic_hash": chunk.semantic_hash,
                            "citation_density": chunk.citation_density,
                            "equation_density": chunk.equation_density,
                            "retrieval_quality_score": chunk.retrieval_quality_score,
                            "coherence_score": chunk.coherence_score,
                            "structural_integrity_score": chunk.structural_integrity_score,
                            "narrative_continuity_score": chunk.narrative_continuity_score,
                            "semantic_boundary_score": chunk.semantic_boundary_score,
                            "citation_spans": [span.model_dump(mode="json") for span in chunk.citation_spans],
                            "citation_entities": chunk.citation_entities,
                            "chunk_topic_signature": chunk.chunk_topic_signature,
                            "previous_chunk_id": chunk.previous_chunk_id,
                            "next_chunk_id": chunk.next_chunk_id,
                            "source_chunk_path": str(source_path),
                        },
                        embedding=vector_list,
                        created_at=utc_now(),
                    )
                    payload_records.append(record)
                    written_records += 1
                    self.manifest.mark(
                        entry_id,
                        source="retrieval.embedding",
                        status="embedded",
                        payload={
                            "model_name": self.model_name,
                            "namespace": namespace_value,
                            "chunk_id": chunk.chunk_id,
                            "paper_id": chunk.paper_id,
                        },
                    )
            if not payload_records:
                continue

            output_path = self._output_path(paper_chunks[0][0], paper_id)
            existing = read_json(output_path, default={"paper_id": paper_id, "model_name": self.model_name, "embeddings": []})
            merged = {item["chunk_id"]: item for item in existing.get("embeddings", [])}
            for record in payload_records:
                merged[record.chunk_id] = record.model_dump(mode="json")
            write_json(
                output_path,
                {
                    "paper_id": paper_id,
                    "model_name": self.model_name,
                    "namespace": namespace_value,
                    "backend": self.embedder.backend,
                    "vector_dim": self.embedder.vector_dim,
                    "embeddings": [merged[key] for key in sorted(merged)],
                },
            )
            output_files.append(str(output_path))

        report = {
            "model_name": self.model_name,
            "model_slug": self.model_slug,
            "namespace": namespace_value,
            "backend": self.embedder.backend,
            "vector_dim": self.embedder.vector_dim,
            "eligible_chunks": len(chunk_records),
            "written_records": written_records,
            "skipped_records": skipped_records,
            "output_files": output_files,
            "generated_at": utc_now(),
        }
        analytics_path = self.analytics.write_report(f"embedding_report_{self.model_slug}", report)
        report["analytics_path"] = str(analytics_path)
        self.logger.info("Generated retrieval embeddings", extra={"context": report})
        return report
