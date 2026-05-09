"""Vector store selection helpers."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.retrieval.vectorstores.faiss_store import FaissVectorStore


def get_vector_store(config: AppConfig, model_name: str, backend: str | None = None):
    resolved = (backend or config.retrieval.vector_db.default_backend).lower()
    if resolved != "faiss":
        raise ValueError(f"Unsupported vector store backend in this runtime: {resolved}")
    return FaissVectorStore(config, model_name)

