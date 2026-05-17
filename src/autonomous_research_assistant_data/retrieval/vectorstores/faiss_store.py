"""FAISS-backed vector store with numpy fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import EmbeddingRecord, VectorIndexRecord
from autonomous_research_assistant_data.retrieval.common import slugify_model_name, stable_hash_text
from autonomous_research_assistant_data.storage.file_store import ensure_directory


class FaissVectorStore:
    """Persist vector indexes locally with optional FAISS acceleration."""

    backend_name = "faiss"

    def __init__(self, config: AppConfig, model_name: str) -> None:
        self.config = config
        self.model_name = model_name
        self.model_slug = slugify_model_name(model_name)
        self.logger = get_logger("retrieval.vectorstores.faiss")
        self._index = None
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._record_map: dict[str, EmbeddingRecord] = {}
        self._faiss = None

    def _base_dir(self, namespace: str) -> Path:
        return self.config.retrieval.vector_indexes_dir / self.backend_name / self.model_slug / namespace

    def _metadata_path(self, namespace: str) -> Path:
        return self._base_dir(namespace) / "metadata.json"

    def _vectors_path(self, namespace: str) -> Path:
        return self._base_dir(namespace) / "vectors.npy"

    def _faiss_path(self, namespace: str) -> Path:
        return self._base_dir(namespace) / "index.faiss"

    def _load_faiss(self) -> None:
        if self._faiss is not None:
            return
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except Exception:
            self._faiss = False

    def build(self, records: list[EmbeddingRecord], *, namespace: str, force_rebuild: bool = False) -> dict[str, Any]:
        base_dir = ensure_directory(self._base_dir(namespace))
        metadata_path = self._metadata_path(namespace)
        vectors_path = self._vectors_path(namespace)
        faiss_path = self._faiss_path(namespace)
        if metadata_path.exists() and not force_rebuild and not self.config.retrieval.vector_db.incremental_updates:
            self.load(namespace=namespace)
            return {
                "status": "loaded_existing",
                "backend": self.backend_name,
                "namespace": namespace,
                "document_count": len(self._record_map),
                "index_path": str(faiss_path if faiss_path.exists() else vectors_path),
            }

        records = sorted(records, key=lambda item: item.chunk_id)
        self._record_map = {record.chunk_id: record for record in records}
        self._vectors = np.asarray([record.embedding for record in records], dtype=np.float32)
        self._load_faiss()
        if getattr(self, "_faiss", None):
            index = self._faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(self._vectors)
            self._faiss.write_index(index, str(faiss_path))
            self._index = index
            backend_runtime = "faiss"
        else:
            np.save(vectors_path, self._vectors)
            self._index = None
            backend_runtime = "numpy"

        metadata_payload = {
            "index": VectorIndexRecord(
                index_id=stable_hash_text(f"{self.model_slug}:{namespace}:{len(records)}"),
                namespace=namespace,
                backend=backend_runtime,
                model_name=self.model_name,
                vector_dim=int(self._vectors.shape[1]) if self._vectors.size else 0,
                document_count=len(records),
                index_path=faiss_path if getattr(self, "_faiss", None) else vectors_path,
                metadata_path=metadata_path,
                created_at=utc_now(),
                updated_at=utc_now(),
                extra={"metric": self.config.retrieval.vector_db.metric, "version": self.config.retrieval.vector_db.version},
            ).model_dump(mode="json"),
            "records": [record.model_dump(mode="json") for record in records],
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, default=str), encoding="utf-8")
        return {
            "status": "built",
            "backend": backend_runtime,
            "namespace": namespace,
            "document_count": len(records),
            "index_path": str(faiss_path if backend_runtime == "faiss" else vectors_path),
            "metadata_path": str(metadata_path),
            "vector_dim": int(self._vectors.shape[1]) if self._vectors.size else 0,
        }

    def load(self, *, namespace: str) -> None:
        metadata_path = self._metadata_path(namespace)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = [EmbeddingRecord.model_validate(record) for record in payload.get("records", [])]
        self._record_map = {record.chunk_id: record for record in records}
        self._load_faiss()
        faiss_path = self._faiss_path(namespace)
        if getattr(self, "_faiss", None) and faiss_path.exists():
            self._index = self._faiss.read_index(str(faiss_path))
            self._vectors = np.asarray([record.embedding for record in records], dtype=np.float32)
        else:
            vectors_path = self._vectors_path(namespace)
            if vectors_path.exists():
                self._vectors = np.load(vectors_path)
            else:
                self._vectors = np.asarray([record.embedding for record in records], dtype=np.float32)
            self._index = None

    def search(self, vector: list[float], *, top_k: int, namespace: str) -> list[tuple[str, float]]:
        if not self._record_map:
            self.load(namespace=namespace)
        query = np.asarray([vector], dtype=np.float32)
        if self._index is not None:
            scores, indices = self._index.search(query, top_k)
            chunk_ids = list(self._record_map.keys())
            return [
                (chunk_ids[index], float(score))
                for score, index in zip(scores[0].tolist(), indices[0].tolist(), strict=True)
                if 0 <= index < len(chunk_ids)
            ]
        scores = np.dot(self._vectors, query[0]) if self._vectors.size else np.asarray([], dtype=np.float32)
        indices = np.argsort(scores)[::-1][:top_k]
        chunk_ids = list(self._record_map.keys())
        return [(chunk_ids[int(index)], float(scores[int(index)])) for index in indices]

    @property
    def records(self) -> dict[str, EmbeddingRecord]:
        return self._record_map
