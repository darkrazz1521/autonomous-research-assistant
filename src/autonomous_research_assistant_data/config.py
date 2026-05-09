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


class PdfProcessingConfig(BaseModel):
    source_pdf_dir: Path
    source_metadata_dir: Path
    front_matter_dir: Path
    extracted_text_dir: Path
    cleaned_text_dir: Path
    repaired_text_dir: Path
    sections_dir: Path
    chunks_dir: Path
    references_dir: Path
    citations_dir: Path
    equation_blocks_dir: Path
    isolated_figures_dir: Path
    isolated_tables_dir: Path
    heading_analysis_dir: Path
    dedup_reports_dir: Path
    repair_reports_dir: Path
    manifests_dir: Path
    validation_dir: Path
    reports_dir: Path
    analytics_dir: Path
    enabled: bool = True
    extraction_backend: str = "pymupdf"
    fallback_backends: list[str] = Field(default_factory=lambda: ["pdfplumber"])
    max_papers_per_run: int = 25
    batch_size: int = 5
    overwrite_existing: bool = False
    chunk_size: int = 450
    chunk_overlap: int = 80
    min_overlap_paragraphs: int = 1
    max_overlap_paragraphs: int = 2
    min_chunk_tokens: int = 120
    max_chunk_tokens: int = 650
    abstract_chunk_max_tokens: int = 220
    repair_overlap_buffer_tokens: int = 140
    paragraph_merge_line_threshold: int = 85
    section_min_confidence: float = 0.45
    layout_aware_default: bool = True
    dedup_strict_default: bool = False
    equation_repair_level: str = "balanced"
    enable_column_reconstruction: bool = True
    cleaning_rules: dict[str, Any] = Field(default_factory=dict)
    validation_rules: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_paths(self) -> "PdfProcessingConfig":
        for field_name in (
            "source_pdf_dir",
            "source_metadata_dir",
            "front_matter_dir",
            "extracted_text_dir",
            "cleaned_text_dir",
            "repaired_text_dir",
            "sections_dir",
            "chunks_dir",
            "references_dir",
            "citations_dir",
            "equation_blocks_dir",
            "isolated_figures_dir",
            "isolated_tables_dir",
            "heading_analysis_dir",
            "dedup_reports_dir",
            "repair_reports_dir",
            "manifests_dir",
            "validation_dir",
            "reports_dir",
            "analytics_dir",
        ):
            value = getattr(self, field_name).expanduser()
            setattr(self, field_name, value)
        return self


class RetrievalEmbeddingConfig(BaseModel):
    default_model: str = "BAAI/bge-base-en-v1.5"
    supported_models: list[str] = Field(
        default_factory=lambda: [
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-base-v2",
            "BAAI/bge-m3",
        ]
    )
    batch_size: int = 16
    normalize_embeddings: bool = True
    max_length: int = 512
    device: str = "auto"
    deterministic_fallback_dim: int = 768
    cache_enabled: bool = True
    min_retrieval_quality_score: float = 0.55
    include_flagged_chunks: bool = False


class RetrievalVectorDbConfig(BaseModel):
    default_backend: str = "faiss"
    supported_backends: list[str] = Field(default_factory=lambda: ["faiss", "qdrant", "lancedb", "chroma"])
    namespace: str = "scientific-corpus"
    metric: str = "cosine"
    rebuild_on_schema_change: bool = True
    incremental_updates: bool = True
    version: str = "v1"


class RetrievalRerankerConfig(BaseModel):
    enabled: bool = True
    default_model: str = "BAAI/bge-reranker-base"
    supported_models: list[str] = Field(
        default_factory=lambda: [
            "BAAI/bge-reranker-base",
            "jinaai/jina-reranker-v1-turbo-en",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ]
    )
    top_k_depth: int = 20
    section_priority_boost: float = 0.08
    citation_sensitivity_boost: float = 0.06


class RetrievalSearchConfig(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 10
    hybrid_fusion: str = "rrf"
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    rrf_k: int = 60
    include_neighbors: bool = True
    neighbor_window: int = 1
    citation_expansion: bool = True
    section_aware_boost: bool = True


class RetrievalEvaluationConfig(BaseModel):
    enabled: bool = True
    default_top_k: int = 10
    default_probe_count: int = 25
    use_manual_probes: bool = True
    use_scifact_when_available: bool = True


class RetrievalConfig(BaseModel):
    embeddings_dir: Path
    vector_indexes_dir: Path
    retrieval_cache_dir: Path
    retrieval_analytics_dir: Path
    rerank_cache_dir: Path
    memory_graph_dir: Path
    retrieval_evaluation_dir: Path
    manifests_dir: Path
    embedding: RetrievalEmbeddingConfig = Field(default_factory=RetrievalEmbeddingConfig)
    vector_db: RetrievalVectorDbConfig = Field(default_factory=RetrievalVectorDbConfig)
    reranker: RetrievalRerankerConfig = Field(default_factory=RetrievalRerankerConfig)
    search: RetrievalSearchConfig = Field(default_factory=RetrievalSearchConfig)
    evaluation: RetrievalEvaluationConfig = Field(default_factory=RetrievalEvaluationConfig)

    @model_validator(mode="after")
    def resolve_paths(self) -> "RetrievalConfig":
        for field_name in (
            "embeddings_dir",
            "vector_indexes_dir",
            "retrieval_cache_dir",
            "retrieval_analytics_dir",
            "rerank_cache_dir",
            "memory_graph_dir",
            "retrieval_evaluation_dir",
            "manifests_dir",
        ):
            value = getattr(self, field_name).expanduser()
            setattr(self, field_name, value)
        return self


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
    pdf_processing: PdfProcessingConfig
    retrieval: RetrievalConfig

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

    for field_name in (
        "source_pdf_dir",
        "source_metadata_dir",
        "front_matter_dir",
        "extracted_text_dir",
        "cleaned_text_dir",
        "repaired_text_dir",
        "sections_dir",
        "chunks_dir",
        "references_dir",
        "citations_dir",
        "equation_blocks_dir",
        "isolated_figures_dir",
        "isolated_tables_dir",
        "heading_analysis_dir",
        "dedup_reports_dir",
        "repair_reports_dir",
        "manifests_dir",
        "validation_dir",
        "reports_dir",
        "analytics_dir",
    ):
        value = getattr(config.pdf_processing, field_name).expanduser()
        if not value.is_absolute():
            value = config.storage.root_dir / value
        setattr(config.pdf_processing, field_name, value)

    for field_name in (
        "embeddings_dir",
        "vector_indexes_dir",
        "retrieval_cache_dir",
        "retrieval_analytics_dir",
        "rerank_cache_dir",
        "memory_graph_dir",
        "retrieval_evaluation_dir",
        "manifests_dir",
    ):
        value = getattr(config.retrieval, field_name).expanduser()
        if not value.is_absolute():
            value = config.storage.root_dir / value
        setattr(config.retrieval, field_name, value)

    return config
