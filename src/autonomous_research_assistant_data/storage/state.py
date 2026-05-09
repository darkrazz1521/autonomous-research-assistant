"""Persistent state store for resumable ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class StateStore:
    """JSON-backed state registry for ingestion checkpoints."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path, default={})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.flush()

    def flush(self) -> None:
        write_json(self.path, self.data)

