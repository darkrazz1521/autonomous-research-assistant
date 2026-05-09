"""Bootstrap helpers for project directory creation."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.storage.file_store import ensure_directory


def bootstrap_directories(config: AppConfig) -> None:
    """Create the expected directory tree for Phase 2."""
    required_dirs = [
        config.storage.datasets_dir,
        config.storage.raw_dir,
        config.storage.processed_dir,
        config.storage.metadata_dir,
        config.storage.external_dir,
        config.storage.state_dir,
        config.storage.logs_dir,
        config.storage.huggingface_cache_dir,
        config.storage.raw_dir / "arxiv" / "pdfs",
        config.storage.raw_dir / "arxiv" / "metadata",
        config.storage.raw_dir / "scifact",
        config.storage.raw_dir / "fever",
        config.storage.raw_dir / "msmarco",
        config.storage.metadata_dir / "manifests",
        config.storage.metadata_dir / "schemas",
        config.storage.logs_dir / "ingestion",
        config.storage.logs_dir / "failed",
    ]
    for directory in required_dirs:
        ensure_directory(directory)

