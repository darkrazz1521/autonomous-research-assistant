"""Bootstrap helpers for project directory creation."""

from __future__ import annotations

import os

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.environment import gpu_runtime_info
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.storage.file_store import ensure_directory
from autonomous_research_assistant_data.utils.colab import maybe_mount_google_drive


def bootstrap_directories(config: AppConfig) -> None:
    """Create the expected directory tree for PDF processing and repair artifacts."""
    required_dirs = [
        config.storage.datasets_dir,
        config.storage.raw_dir,
        config.storage.processed_dir,
        config.storage.metadata_dir,
        config.storage.external_dir,
        config.storage.state_dir,
        config.storage.logs_dir,
        config.storage.huggingface_cache_dir,
        config.storage.huggingface_home_dir,
        config.storage.raw_dir / "arxiv" / "pdfs",
        config.storage.raw_dir / "arxiv" / "metadata",
        config.storage.raw_dir / "scifact",
        config.storage.raw_dir / "fever",
        config.storage.raw_dir / "msmarco",
        config.storage.metadata_dir / "manifests",
        config.storage.metadata_dir / "schemas",
        config.storage.logs_dir / "ingestion",
        config.storage.logs_dir / "failed",
        config.pdf_processing.extracted_text_dir,
        config.pdf_processing.cleaned_text_dir,
        config.pdf_processing.repaired_text_dir,
        config.pdf_processing.sections_dir,
        config.pdf_processing.chunks_dir,
        config.pdf_processing.front_matter_dir,
        config.pdf_processing.references_dir,
        config.pdf_processing.citations_dir,
        config.pdf_processing.equation_blocks_dir,
        config.pdf_processing.isolated_figures_dir,
        config.pdf_processing.isolated_tables_dir,
        config.pdf_processing.heading_analysis_dir,
        config.pdf_processing.dedup_reports_dir,
        config.pdf_processing.repair_reports_dir,
        config.pdf_processing.manifests_dir,
        config.pdf_processing.validation_dir,
        config.pdf_processing.reports_dir,
        config.pdf_processing.analytics_dir,
        config.retrieval.embeddings_dir,
        config.retrieval.vector_indexes_dir,
        config.retrieval.retrieval_cache_dir,
        config.retrieval.retrieval_analytics_dir,
        config.retrieval.rerank_cache_dir,
        config.retrieval.memory_graph_dir,
        config.retrieval.retrieval_evaluation_dir,
        config.retrieval.manifests_dir,
        config.rag.rag_cache_dir,
        config.rag.rag_outputs_dir,
        config.rag.research_sessions_dir,
        config.rag.rag_evaluation_dir,
        config.rag.generated_answers_dir,
    ]
    for directory in required_dirs:
        if directory is not None:
            ensure_directory(directory)


def prepare_runtime(config: AppConfig) -> None:
    """Prepare environment-specific runtime resources."""
    logger = get_logger("runtime.bootstrap")
    if config.storage.huggingface_home_dir is not None:
        os.environ.setdefault("HF_HOME", str(config.storage.huggingface_home_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(config.storage.huggingface_cache_dir))
    if config.huggingface.offline_mode:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if config.colab.enabled:
        maybe_mount_google_drive(
            enabled=config.colab.mount_google_drive,
            mount_point=config.colab.google_drive_mount_point,
        )
    logger.info(
        "Prepared runtime",
        extra={
            "context": {
                "environment": config.runtime.environment,
                "gpu": gpu_runtime_info(),
                "google_drive_mount_point": str(config.colab.google_drive_mount_point),
                "hf_home": os.environ.get("HF_HOME"),
                "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
            }
        },
    )
