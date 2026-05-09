"""Shared helpers for arXiv ingestion modes."""

from __future__ import annotations

from pathlib import Path

from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ArxivPaperRecord
from autonomous_research_assistant_data.storage.manifest import ManifestStore
from autonomous_research_assistant_data.storage.metadata_store import MetadataStore
from autonomous_research_assistant_data.storage.state import StateStore


class ArxivStorageManager:
    """Shared arXiv storage and manifest utilities."""

    def __init__(
        self,
        pdf_root: Path,
        metadata_root: Path,
        metadata_store: MetadataStore,
        manifest_store: ManifestStore,
        state_store: StateStore,
        verify_existing_files: bool = True,
    ) -> None:
        self.pdf_root = pdf_root
        self.metadata_root = metadata_root
        self.metadata_store = metadata_store
        self.manifest_store = manifest_store
        self.state_store = state_store
        self.verify_existing_files = verify_existing_files

    def safe_id(self, arxiv_id: str) -> str:
        """Convert arXiv ids to filesystem-safe names."""
        return arxiv_id.replace("/", "_")

    def paper_paths(self, paper: ArxivPaperRecord) -> tuple[Path, Path]:
        """Build stable PDF and metadata storage paths."""
        year = str(paper.published_at.year)
        month = f"{paper.published_at.month:02d}"
        safe_id = self.safe_id(paper.arxiv_id)
        pdf_path = self.pdf_root / year / month / f"{safe_id}.pdf"
        metadata_path = self.metadata_root / year / month / f"{safe_id}.json"
        return pdf_path, metadata_path

    def already_ingested(self, paper: ArxivPaperRecord) -> bool:
        """Check whether a paper already exists in the manifest and on disk."""
        pdf_path, metadata_path = self.paper_paths(paper)
        manifest_hit = self.manifest_store.exists(paper.arxiv_id)
        if not self.verify_existing_files:
            return manifest_hit
        return manifest_hit and pdf_path.exists() and metadata_path.exists()

    def persist_record(self, paper: ArxivPaperRecord, status: str) -> None:
        """Persist metadata and register the paper in the manifest."""
        pdf_path, metadata_path = self.paper_paths(paper)
        paper.pdf_path = pdf_path
        paper.metadata_path = metadata_path
        paper.download_timestamp = utc_now()
        self.metadata_store.save_arxiv_record(metadata_path, paper)
        self.manifest_store.mark(
            paper.arxiv_id,
            source="arxiv",
            status=status,
            payload={
                "title": paper.title,
                "authors": paper.authors,
                "categories": paper.categories,
                "published_at": paper.published_at.isoformat(),
                "updated_at": paper.updated_at.isoformat(),
                "pdf_url": paper.pdf_url,
                "pdf_path": str(pdf_path),
                "metadata_path": str(metadata_path),
            },
        )

    def mark_failure(self, paper: ArxivPaperRecord, error: str) -> None:
        """Register a failed paper in the manifest."""
        self.manifest_store.mark(
            paper.arxiv_id,
            source="arxiv",
            status="failed",
            payload={"error": error, "pdf_url": paper.pdf_url},
        )

    def update_watermark(self, latest_updated_at: str | None) -> None:
        """Persist the latest incremental sync checkpoints."""
        if latest_updated_at:
            self.state_store.set("arxiv.last_updated_at", latest_updated_at)
        self.state_store.set("arxiv.last_run_at", utc_now().isoformat())

