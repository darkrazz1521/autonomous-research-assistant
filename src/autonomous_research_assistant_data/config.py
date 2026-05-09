"""Application configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class StorageConfig(BaseModel):
    root_dir: Path
    datasets_dir: Path
    raw_dir: Path
    processed_dir: Path
    metadata_dir: Path
    external_dir: Path
    state_dir: Path
    logs_dir: Path
    huggingface_cache_dir: Path
    parquet_export_enabled: bool = True
    local_source_code_only: bool = False

    @model_validator(mode="after")
    def resolve_paths(self) -> "StorageConfig":
        for field_name in (
            "root_dir",
            "datasets_dir",
            "raw_dir",
            "processed_dir",
            "metadata_dir",
            "external_dir",
            "state_dir",
            "logs_dir",
            "huggingface_cache_dir",
        ):
            value = getattr(self, field_name)
            setattr(self, field_name, value.expanduser())
        return self


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console_enabled: bool = True
    ingestion_log_file: Path
    failure_log_file: Path


class RetryConfig(BaseModel):
    max_attempts: int = 4
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    backoff_multiplier: float = 2.0
    jitter_seconds: float = 0.25


class ArxivConfig(BaseModel):
    enabled: bool = True
    base_url: str
    categories: list[str]
    batch_size: int = 100
    max_api_results_per_run: int = 500
    max_pdf_downloads_per_run: int = 250
    pdf_download_concurrency: int = 8
    request_timeout_seconds: int = 30
    download_timeout_seconds: int = 120
    request_pause_seconds: float = 0.5
    user_agent: str
    incremental_lookback_days: int = 2
    save_metadata_format: str = "json"
    verify_existing_files: bool = True


class DatasetIngestionConfig(BaseModel):
    enabled: bool = True
    loader: str = "huggingface"
    huggingface_dataset_id: str
    configs: list[str] = Field(default_factory=list)
    split_overrides: dict[str, str] = Field(default_factory=dict)
    trust_remote_code: bool = False
    export_format: str = "parquet"
    export_enabled: bool = True


class ValidationConfig(BaseModel):
    require_non_empty_splits: bool = True
    verify_pdf_files: bool = True
    verify_metadata_files: bool = True
    fail_on_missing_manifest_entries: bool = False


class AppConfig(BaseModel):
    project_name: str
    profile: str
    storage: StorageConfig
    logging: LoggingConfig
    retry: RetryConfig
    arxiv: ArxivConfig
    datasets: dict[str, DatasetIngestionConfig]
    validation: ValidationConfig


def load_config(config_path: str | Path) -> AppConfig:
    """Load application settings from a YAML file."""
    path = Path(config_path)
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(payload)

