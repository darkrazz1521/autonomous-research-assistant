"""Prepare graph-ready retrieval memory metadata."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import EmbeddingRecord, NeighborRecord
from autonomous_research_assistant_data.retrieval.common import slugify_model_name
from autonomous_research_assistant_data.retrieval.vectorstores.faiss_store import FaissVectorStore
from autonomous_research_assistant_data.storage.file_store import write_json


class MemoryGraphBuilder:
    """Generate graph-ready chunk neighbor metadata."""

    def __init__(self, config: AppConfig, model_name: str, store: FaissVectorStore) -> None:
        self.config = config
        self.model_name = model_name
        self.model_slug = slugify_model_name(model_name)
        self.store = store

    def _path(self, namespace: str) -> Path:
        return self.config.retrieval.memory_graph_dir / self.model_slug / namespace / "memory_graph.json"

    def build(self, *, namespace: str) -> dict:
        records = list(self.store.records.values())
        neighbors: list[NeighborRecord] = []
        by_paper: dict[str, list[EmbeddingRecord]] = defaultdict(list)
        for record in records:
            by_paper[record.paper_id].append(record)
            if record.metadata.get("previous_chunk_id"):
                neighbors.append(
                    NeighborRecord(
                        source_chunk_id=record.chunk_id,
                        target_chunk_id=str(record.metadata["previous_chunk_id"]),
                        relation_type="section_neighbor",
                        score=1.0,
                    )
                )
            if record.metadata.get("next_chunk_id"):
                neighbors.append(
                    NeighborRecord(
                        source_chunk_id=record.chunk_id,
                        target_chunk_id=str(record.metadata["next_chunk_id"]),
                        relation_type="section_neighbor",
                        score=1.0,
                    )
                )

        citation_groups: dict[str, list[str]] = defaultdict(list)
        for record in records:
            for entity in record.metadata.get("citation_entities", []):
                citation_groups[str(entity)].append(record.chunk_id)
        for entity, chunk_ids in citation_groups.items():
            if len(chunk_ids) < 2:
                continue
            for source in chunk_ids:
                for target in chunk_ids:
                    if source == target:
                        continue
                    neighbors.append(
                        NeighborRecord(
                            source_chunk_id=source,
                            target_chunk_id=target,
                            relation_type="citation_neighbor",
                            score=0.8,
                            metadata={"entity": entity},
                        )
                    )

        for record in records:
            hits = self.store.search(record.embedding, top_k=3, namespace=namespace)
            for chunk_id, score in hits:
                if chunk_id == record.chunk_id:
                    continue
                neighbors.append(
                    NeighborRecord(
                        source_chunk_id=record.chunk_id,
                        target_chunk_id=chunk_id,
                        relation_type="semantic_neighbor",
                        score=round(float(score), 6),
                    )
                )

        payload = {
            "model_name": self.model_name,
            "namespace": namespace,
            "generated_at": utc_now(),
            "neighbor_count": len(neighbors),
            "paper_neighbors": {paper_id: [item.chunk_id for item in paper_records] for paper_id, paper_records in by_paper.items()},
            "neighbors": [neighbor.model_dump(mode="json") for neighbor in neighbors],
        }
        path = self._path(namespace)
        write_json(path, payload)
        return {"path": str(path), "neighbor_count": len(neighbors)}

