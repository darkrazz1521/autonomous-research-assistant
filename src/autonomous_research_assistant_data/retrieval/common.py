"""Shared retrieval helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.models.common import ChunkRecord, EmbeddingRecord


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify_model_name(model_name: str) -> str:
    slug = model_name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def chunk_sort_key(chunk: ChunkRecord) -> tuple[str, int, int, int]:
    return (chunk.paper_id, chunk.section_index, chunk.chunk_index, chunk.page_range[0] or 0)


def iter_chunk_payloads(chunks_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(chunks_dir.rglob("*.json")):
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return payloads


def load_chunk_records(chunks_dir: Path) -> list[tuple[Path, ChunkRecord]]:
    records: list[tuple[Path, ChunkRecord]] = []
    for path, payload in iter_chunk_payloads(chunks_dir):
        for chunk_payload in payload.get("chunks", []):
            records.append((path, ChunkRecord.model_validate(chunk_payload)))
    records.sort(key=lambda item: chunk_sort_key(item[1]))
    return records


def load_chunk_record_map(chunks_dir: Path) -> dict[str, tuple[Path, ChunkRecord]]:
    return {chunk.chunk_id: (path, chunk) for path, chunk in load_chunk_records(chunks_dir)}


def load_embedding_records(embeddings_dir: Path, model_name: str) -> list[EmbeddingRecord]:
    model_dir = embeddings_dir / slugify_model_name(model_name)
    records: list[EmbeddingRecord] = []
    if not model_dir.exists():
        return records
    for path in sorted(model_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record_payload in payload.get("embeddings", []):
            records.append(EmbeddingRecord.model_validate(record_payload))
    return records
