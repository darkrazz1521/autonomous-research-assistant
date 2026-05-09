"""Manifest storage for duplicate detection and run bookkeeping."""

from __future__ import annotations

from pathlib import Path

from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ManifestEntry
from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class ManifestStore:
    """Simple JSON manifest store keyed by entry id."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path, default={"entries": {}})

    def exists(self, entry_id: str) -> bool:
        return entry_id in self.data["entries"]

    def get(self, entry_id: str) -> dict | None:
        return self.data["entries"].get(entry_id)

    def upsert(self, entry: ManifestEntry) -> None:
        self.data["entries"][entry.entry_id] = entry.model_dump(mode="json")
        self.flush()

    def mark(
        self,
        entry_id: str,
        source: str,
        status: str,
        payload: dict,
    ) -> None:
        now = utc_now()
        existing = self.data["entries"].get(entry_id)
        created_at = existing["created_at"] if existing else now
        entry = ManifestEntry(
            entry_id=entry_id,
            source=source,
            status=status,
            created_at=created_at,
            updated_at=now,
            payload=payload,
        )
        self.upsert(entry)

    def flush(self) -> None:
        write_json(self.path, self.data)

