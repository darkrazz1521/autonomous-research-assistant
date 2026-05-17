"""Vector index build pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Any

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.retrieval.analytics.reporter import RetrievalAnalyticsReporter
from autonomous_research_assistant_data.retrieval.common import load_embedding_records, slugify_model_name
from autonomous_research_assistant_data.retrieval.memory.builder import MemoryGraphBuilder
from autonomous_research_assistant_data.retrieval.vectorstores.registry import get_vector_store
from autonomous_research_assistant_data.storage.manifest import ManifestStore


class VectorIndexBuilder:
    """Build persistent vector indexes from embedding artifacts."""

    def __init__(self, config: AppConfig, *, model_name: str | None = None, backend: str | None = None) -> None:
        self.config = config
        self.model_name = model_name or config.retrieval.embedding.default_model
        self.backend = backend or config.retrieval.vector_db.default_backend
        self.model_slug = slugify_model_name(self.model_name)
        self.logger = get_logger("retrieval.vectorstores.builder")
        self.manifest = ManifestStore(config.retrieval.manifests_dir / "vector_index_manifest.json")
        self.analytics = RetrievalAnalyticsReporter(config.retrieval.retrieval_analytics_dir)

    def build(self, *, force_rebuild: bool = False, namespace: str | None = None, build_memory_graph: bool = True) -> dict[str, Any]:
        namespace_value = namespace or self.config.retrieval.vector_db.namespace
        records = load_embedding_records(self.config.retrieval.embeddings_dir, self.model_name)
        if not records:
            raise FileNotFoundError(
                f"No embeddings found for model {self.model_name} under {self.config.retrieval.embeddings_dir}. "
                "Run generate_embeddings after processed chunks are available."
            )

        store = get_vector_store(self.config, self.model_name, self.backend)
        stats = store.build(records, namespace=namespace_value, force_rebuild=force_rebuild)
        duplicate_vectors = sum(count - 1 for count in Counter(tuple(record.embedding[:32]) for record in records).values() if count > 1)
        analytics = {
            "model_name": self.model_name,
            "backend": stats["backend"],
            "namespace": namespace_value,
            "document_count": len(records),
            "vector_dim": stats["vector_dim"],
            "embedding_coverage": 1.0,
            "duplicate_vectors": duplicate_vectors,
            "vector_density": round(sum(sum(abs(value) > 1e-8 for value in record.embedding) for record in records) / max(len(records) * stats["vector_dim"], 1), 6),
            "generated_at": utc_now(),
            "index_path": stats["index_path"],
            "metadata_path": stats["metadata_path"],
        }
        analytics_path = self.analytics.write_report(f"vector_index_{self.model_slug}_{namespace_value}", analytics)
        self.manifest.mark(
            f"{self.model_slug}:{namespace_value}",
            source="retrieval.vector_index",
            status="built",
            payload={
                "model_name": self.model_name,
                "backend": stats["backend"],
                "namespace": namespace_value,
                "document_count": len(records),
                "analytics_path": str(analytics_path),
            },
        )

        memory_graph = None
        if build_memory_graph:
            store.load(namespace=namespace_value)
            memory_graph = MemoryGraphBuilder(self.config, self.model_name, store).build(namespace=namespace_value)

        payload = {
            **stats,
            "analytics_path": str(analytics_path),
            "memory_graph": memory_graph,
        }
        self.logger.info("Built vector index", extra={"context": payload})
        return payload
