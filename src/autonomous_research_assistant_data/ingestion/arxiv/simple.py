"""Conservative arXiv ingestion mode for validation and Colab-safe runs."""

from __future__ import annotations

import time

import arxiv
from tqdm import tqdm

from autonomous_research_assistant_data.core.retry import retry_sync
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.ingestion.arxiv.common import ArxivStorageManager
from autonomous_research_assistant_data.ingestion.base import BaseIngestor, IngestionContext
from autonomous_research_assistant_data.models.common import ArxivPaperRecord
from autonomous_research_assistant_data.storage.file_store import append_jsonl, ensure_directory


class SimpleArxivIngestor(BaseIngestor):
    """Sequential arXiv ingestion using the official Python package."""

    source_name = "arxiv.simple"

    def __init__(self, context: IngestionContext) -> None:
        super().__init__(context)
        self.simple_config = self.config.arxiv.simple_mode
        self.failure_log = self.config.logging.failure_log_file
        self.storage_manager = ArxivStorageManager(
            pdf_root=self.config.storage.raw_dir / "arxiv" / "pdfs",
            metadata_root=self.config.storage.raw_dir / "arxiv" / "metadata",
            metadata_store=self.context.metadata_store,
            manifest_store=self.context.manifest_store,
            state_store=self.context.state_store,
            verify_existing_files=self.config.arxiv.verify_existing_files,
        )

    def _build_query(self) -> str:
        return " OR ".join(f"cat:{category}" for category in self.config.arxiv.categories)

    def _build_client(self) -> arxiv.Client:
        return arxiv.Client(
            page_size=self.simple_config.search_page_size,
            delay_seconds=max(5.0, self.simple_config.delay_between_downloads_seconds),
            num_retries=min(self.config.retry.max_attempts, 2),
        )

    def _sort_by(self) -> arxiv.SortCriterion:
        mapping = {
            "submitteddate": arxiv.SortCriterion.SubmittedDate,
            "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
            "relevance": arxiv.SortCriterion.Relevance,
        }
        return mapping.get(self.simple_config.sort_by.lower(), arxiv.SortCriterion.SubmittedDate)

    def _sort_order(self) -> arxiv.SortOrder:
        return arxiv.SortOrder.Descending if self.simple_config.sort_order.lower() == "descending" else arxiv.SortOrder.Ascending

    def _result_to_record(self, result: arxiv.Result) -> ArxivPaperRecord:
        pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{result.get_short_id()}.pdf"
        return ArxivPaperRecord(
            arxiv_id=result.get_short_id(),
            title=(result.title or "").strip(),
            authors=[author.name for author in result.authors],
            abstract=" ".join((result.summary or "").split()),
            categories=list(result.categories),
            published_at=result.published,
            updated_at=result.updated,
            pdf_url=pdf_url,
        )

    def _build_search(self) -> arxiv.Search:
        search_max_results = max(self.simple_config.search_page_size, self.simple_config.max_results * 3)
        return arxiv.Search(
            query=self._build_query(),
            max_results=search_max_results,
            sort_by=self._sort_by(),
            sort_order=self._sort_order(),
        )

    def _download_pdf(self, result: arxiv.Result, paper: ArxivPaperRecord) -> None:
        pdf_path, _ = self.storage_manager.paper_paths(paper)
        ensure_directory(pdf_path.parent)

        def _save() -> None:
            result.download_pdf(dirpath=str(pdf_path.parent), filename=pdf_path.name)

        retry_sync(_save, self.config.retry, (Exception,))

    def ingest(self) -> dict[str, int]:
        stats = {"candidates": 0, "downloaded": 0, "skipped": 0, "failed": 0}
        saved_records: list[dict] = []
        latest_updated_at: str | None = None
        last_updated_at = self.context.state_store.get("arxiv.last_updated_at")

        progress = tqdm(self._build_client().results(self._build_search()), total=self.simple_config.max_results, desc="Simple arXiv ingestion")
        for result in progress:
            paper = self._result_to_record(result)

            if last_updated_at and paper.updated_at.isoformat() <= last_updated_at:
                continue
            if self.simple_config.resume_from_manifest and self.storage_manager.already_ingested(paper):
                stats["skipped"] += 1
                continue

            stats["candidates"] += 1
            if stats["candidates"] > self.simple_config.max_results:
                break

            try:
                if not self.simple_config.metadata_only:
                    self._download_pdf(result, paper)
                    time.sleep(self.simple_config.delay_between_downloads_seconds)
                self.storage_manager.persist_record(
                    paper,
                    status="metadata_only" if self.simple_config.metadata_only else "downloaded",
                )
                saved_records.append(paper.model_dump(mode="json"))
                stats["downloaded"] += 1
                if latest_updated_at is None or paper.updated_at.isoformat() > latest_updated_at:
                    latest_updated_at = paper.updated_at.isoformat()
            except Exception as exc:
                append_jsonl(
                    self.failure_log,
                    {
                        "source": "arxiv_simple",
                        "arxiv_id": paper.arxiv_id,
                        "pdf_url": paper.pdf_url,
                        "error": str(exc),
                        "timestamp": utc_now().isoformat(),
                    },
                )
                self.storage_manager.mark_failure(paper, str(exc))
                self.logger.warning(
                    "Failed simple arXiv item",
                    extra={"context": {"arxiv_id": paper.arxiv_id, "error": str(exc)}},
                )
                stats["failed"] += 1

        if saved_records and self.config.storage.parquet_export_enabled:
            summary_path = self.config.storage.metadata_dir / "arxiv_records_simple_latest.parquet"
            self.context.metadata_store.export_records_to_parquet(summary_path, saved_records)

        self.storage_manager.update_watermark(latest_updated_at)
        self.logger.info("Completed simple arXiv ingestion", extra={"context": stats})
        return stats
