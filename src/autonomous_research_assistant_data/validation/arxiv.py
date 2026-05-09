"""Validation checks for arXiv storage and metadata."""

from __future__ import annotations

from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.storage.file_store import read_json


class ArxivValidator:
    """Validate raw arXiv artifacts and manifest consistency."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def validate(self) -> dict[str, int]:
        manifest_path = self.config.storage.metadata_dir / "manifests" / "arxiv_manifest.json"
        manifest = read_json(manifest_path, default={"entries": {}})
        entries = manifest.get("entries", {})

        missing_pdf = 0
        missing_metadata = 0
        valid = 0

        for payload in entries.values():
            item = payload.get("payload", {})
            pdf_path = Path(item.get("pdf_path", ""))
            metadata_path = Path(item.get("metadata_path", ""))
            if self.config.validation.verify_pdf_files and pdf_path and not pdf_path.exists():
                missing_pdf += 1
                continue
            if self.config.validation.verify_metadata_files and metadata_path and not metadata_path.exists():
                missing_metadata += 1
                continue
            valid += 1
        return {
            "entries": len(entries),
            "valid": valid,
            "missing_pdf": missing_pdf,
            "missing_metadata": missing_metadata,
        }

