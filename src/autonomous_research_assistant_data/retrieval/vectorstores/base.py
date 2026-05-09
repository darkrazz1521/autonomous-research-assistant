"""Vector store interfaces."""

from __future__ import annotations

from typing import Protocol

from autonomous_research_assistant_data.models.common import EmbeddingRecord


class VectorStore(Protocol):
    backend_name: str

    def build(self, records: list[EmbeddingRecord], *, namespace: str, force_rebuild: bool = False) -> dict: ...

    def load(self, *, namespace: str) -> None: ...

    def search(self, vector: list[float], *, top_k: int, namespace: str) -> list[tuple[str, float]]: ...

