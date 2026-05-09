"""Base classes for ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.storage.file_store import ensure_directory
from autonomous_research_assistant_data.storage.manifest import ManifestStore
from autonomous_research_assistant_data.storage.metadata_store import MetadataStore
from autonomous_research_assistant_data.storage.state import StateStore


@dataclass
class IngestionContext:
    config: AppConfig
    metadata_store: MetadataStore
    state_store: StateStore
    manifest_store: ManifestStore


class BaseIngestor:
    """Base ingestion unit with common path and logger helpers."""

    source_name: str

    def __init__(self, context: IngestionContext) -> None:
        self.context = context
        self.config = context.config
        self.logger = get_logger(f"ingestion.{self.source_name}")

    def ensure(self, path: Path) -> Path:
        return ensure_directory(path)

