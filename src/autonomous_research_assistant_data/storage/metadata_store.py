"""Metadata persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autonomous_research_assistant_data.models.common import ArxivPaperRecord, DatasetArtifactRecord
from autonomous_research_assistant_data.storage.file_store import write_json


class MetadataStore:
    """Persist source metadata records to JSON and optional parquet summaries."""

    def save_arxiv_record(self, path: Path, record: ArxivPaperRecord) -> None:
        write_json(path, record.model_dump(mode="json"))

    def save_dataset_record(self, path: Path, record: DatasetArtifactRecord) -> None:
        write_json(path, record.model_dump(mode="json"))

    def export_records_to_parquet(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame.to_parquet(path, index=False)

