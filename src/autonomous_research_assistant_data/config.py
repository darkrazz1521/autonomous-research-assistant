"""Application configuration management with layered environments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from autonomous_research_assistant_data.core.environment import detect_runtime_environment

EnvironmentName = Literal["local", "colab"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class RuntimeConfig(BaseModel):
    environment: EnvironmentName = "local"
    auto_detect: bool = True


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
    huggingface_home_dir: Path | None = None
    parquet_export_enabled: bool = True
    local_source_code_only: bool = False

    @model_validator(mode="after")
    def resolve_paths(self) -> "StorageConfig":
        self.root_dir = self.root_dir.expanduser()
        for field_name in (
            "datasets_dir",
            "raw_dir",
            "processed_dir",
            "metadata_dir",
            "external_dir",
            "state_dir",
            "logs_dir",
            "huggingface_cache_dir",
            "huggingface_home_dir",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = value.expanduser()
            if not value.is_absolute():
                value = self.root_dir / value
            setattr(self, field_name, value)
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


class HuggingFaceConfig(BaseModel):
    datasets_version: str = "2.19.1"
    supported_datasets_major_minor: str = "2.19"
    hub_version: str = "0.23.4"
    token_env_var: str = "HF_TOKEN"
    enable_version_guard: bool = True
    log_future_warnings: bool = True
    offline_mode: bool = False


class ColabConfig(BaseModel):
    enabled: bool = False
    mount_google_drive: bool = False
    google_drive_mount_point: Path = Path("/content/drive")
    google_drive_project_dir: Path = Path("/content/drive/MyDrive/autonomous_research_assistant")
    prefer_drive_storage: bool = False
    require_gpu: bool = False


class ArxivConfig(BaseModel):
    class SimpleModeConfig(BaseModel):
        enabled: bool = True
        max_results: int = 15
        delay_between_downloads_seconds: float = 7.0
        search_page_size: int = 20
        sort_by: str = "submittedDate"
        sort_order: str = "descending"
        metadata_only: bool = False
        resume_from_manifest: bool = True

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
    simple_mode: SimpleModeConfig = Field(default_factory=SimpleModeConfig)


class DatasetIngestionConfig(BaseModel):
    enabled: bool = True
    loader: str = "huggingface"
    huggingface_dataset_id: str
    configs: list[str] = Field(default_factory=list)
    split_overrides: dict[str, str] = Field(default_factory=dict)
    revision: str | None = None
    trust_remote_code: bool = False
    allow_legacy_script_fallback: bool = True
    streaming: bool = False
    data_dir: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    export_format: str = "parquet"
    export_enabled: bool = True
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    require_non_empty_splits: bool = True
    verify_pdf_files: bool = True
    verify_metadata_files: bool = True
    fail_on_missing_manifest_entries: bool = False


class AppConfig(BaseModel):
    project_name: str
    runtime: RuntimeConfig
    storage: StorageConfig
    logging: LoggingConfig
    retry: RetryConfig
    huggingface: HuggingFaceConfig
    colab: ColabConfig
    arxiv: ArxivConfig
    datasets: dict[str, DatasetIngestionConfig]
    validation: ValidationConfig

    @property
    def profile(self) -> str:
        return self.runtime.environment


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_environment(environment: str | None) -> EnvironmentName:
    if environment and environment != "auto":
        return environment  # type: ignore[return-value]
    return detect_runtime_environment()


def load_config(
    config_path: str | Path | None = None,
    environment: str | None = None,
    config_dir: str | Path = "configs",
) -> AppConfig:
    """Load configuration using base + environment layering and an optional override file."""
    config_dir_path = Path(config_dir)
    env_name = _resolve_environment(environment)

    payload = _load_yaml(config_dir_path / "base.yaml")
    payload = _deep_merge(payload, _load_yaml(config_dir_path / f"{env_name}.yaml"))

    if config_path:
        payload = _deep_merge(payload, _load_yaml(Path(config_path)))

    payload.setdefault("runtime", {})
    payload["runtime"]["environment"] = env_name
    config = AppConfig.model_validate(payload)

    for field_name in ("ingestion_log_file", "failure_log_file"):
        value = getattr(config.logging, field_name).expanduser()
        if not value.is_absolute():
            value = config.storage.root_dir / value
        setattr(config.logging, field_name, value)

    return config
