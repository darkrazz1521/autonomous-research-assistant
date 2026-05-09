"""Filesystem helpers for persistent storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    """Create a directory path if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk with stable formatting."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a JSONL record."""
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a JSON object from disk, returning a default when absent."""
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))

