"""Shared data models used across ingestion sources."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ArxivPaperRecord(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: datetime
    updated_at: datetime
    pdf_url: str
    pdf_path: Path | None = None
    metadata_path: Path | None = None
    source: str = "arxiv"
    download_timestamp: datetime | None = None


class DatasetArtifactRecord(BaseModel):
    dataset_name: str
    source_type: str = "huggingface"
    source_id: str
    config_name: str | None = None
    split: str
    num_rows: int
    storage_path: Path | None = None
    fingerprint: str | None = None
    downloaded_at: datetime
    extra: dict[str, Any] = Field(default_factory=dict)


class ManifestEntry(BaseModel):
    entry_id: str
    source: str
    status: str
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any]

