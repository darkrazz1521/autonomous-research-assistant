"""Validation checks for benchmark datasets."""

from __future__ import annotations

from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.storage.file_store import read_json


class DatasetValidator:
    """Validate exported dataset artifacts and split metadata."""

    def __init__(self, config: AppConfig, dataset_name: str) -> None:
        self.config = config
        self.dataset_name = dataset_name

    def validate(self) -> dict[str, int]:
        base_metadata_dir = self.config.storage.metadata_dir / self.dataset_name
        metadata_files = list(base_metadata_dir.rglob("*.json"))
        total = 0
        missing = 0
        empty = 0

        for metadata_file in metadata_files:
            total += 1
            payload = read_json(metadata_file)
            path = Path(payload.get("storage_path", ""))
            if not path.exists():
                missing += 1
                continue
            if self.config.validation.require_non_empty_splits and int(payload.get("num_rows", 0)) <= 0:
                empty += 1
        return {"artifacts": total, "missing": missing, "empty": empty}

