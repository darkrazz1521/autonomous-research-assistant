"""High-level arXiv ingestion pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from tqdm.asyncio import tqdm

from autonomous_research_assistant_data.core.retry import retry_async
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.ingestion.arxiv.client import ArxivClient
from autonomous_research_assistant_data.ingestion.base import BaseIngestor, IngestionContext
from autonomous_research_assistant_data.models.common import ArxivPaperRecord
from autonomous_research_assistant_data.storage.file_store import append_jsonl, ensure_directory


class ArxivIngestor(BaseIngestor):
    """Collect arXiv metadata and PDFs with resumable incremental sync."""

    source_name = "arxiv"

    def __init__(self, context: IngestionContext) -> None:
        super().__init__(context)
        self.client = ArxivClient(context.config)
        self.pdf_root = self.config.storage.raw_dir / "arxiv" / "pdfs"
        self.metadata_root = self.config.storage.raw_dir / "arxiv" / "metadata"
        self.failure_log = self.config.logging.failure_log_file

    def _safe_id(self, arxiv_id: str) -> str:
        return arxiv_id.replace("/", "_")

    def _paper_paths(self, paper: ArxivPaperRecord) -> tuple[Path, Path]:
        year = str(paper.published_at.year)
        month = f"{paper.published_at.month:02d}"
        safe_id = self._safe_id(paper.arxiv_id)
        pdf_path = self.pdf_root / year / month / f"{safe_id}.pdf"
        metadata_path = self.metadata_root / year / month / f"{safe_id}.json"
        return pdf_path, metadata_path

    async def collect_candidates(self) -> list[ArxivPaperRecord]:
        max_results = self.config.arxiv.max_api_results_per_run
        batch_size = self.config.arxiv.batch_size
        last_updated_at = self.context.state_store.get("arxiv.last_updated_at")
        watermark = self.client.watermark(last_updated_at)
        collected: list[ArxivPaperRecord] = []

        async with httpx.AsyncClient() as client:
            for start in range(0, max_results, batch_size):
                batch = await self.client.fetch_batch(client, start=start, max_results=batch_size)
                if not batch:
                    break
                stop = False
                for paper in batch:
                    if paper.updated_at <= watermark:
                        stop = True
                        break
                    collected.append(paper)
                if stop:
                    break
        self.logger.info(
            "Collected arXiv candidates",
            extra={"context": {"candidates": len(collected), "watermark": watermark.isoformat()}},
        )
        return collected

    def _already_ingested(self, paper: ArxivPaperRecord) -> bool:
        pdf_path, metadata_path = self._paper_paths(paper)
        manifest_hit = self.context.manifest_store.exists(paper.arxiv_id)
        if not self.config.arxiv.verify_existing_files:
            return manifest_hit
        return manifest_hit and pdf_path.exists() and metadata_path.exists()

    async def _download_pdf(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        paper: ArxivPaperRecord,
    ) -> dict[str, Any]:
        pdf_path, metadata_path = self._paper_paths(paper)
        ensure_directory(pdf_path.parent)
        ensure_directory(metadata_path.parent)

        if self._already_ingested(paper):
            return {"paper": paper, "status": "skipped", "pdf_path": pdf_path, "metadata_path": metadata_path}

        async with semaphore:
            async def _request() -> httpx.Response:
                response = await client.get(
                    paper.pdf_url,
                    headers={"User-Agent": self.config.arxiv.user_agent},
                    timeout=self.config.arxiv.download_timeout_seconds,
                    follow_redirects=True,
                )
                response.raise_for_status()
                return response

            try:
                response = await retry_async(_request, self.config.retry, (httpx.HTTPError,))
                pdf_path.write_bytes(response.content)
                return {"paper": paper, "status": "downloaded", "pdf_path": pdf_path, "metadata_path": metadata_path}
            except Exception as exc:
                append_jsonl(
                    self.failure_log,
                    {
                        "source": "arxiv",
                        "arxiv_id": paper.arxiv_id,
                        "pdf_url": paper.pdf_url,
                        "error": str(exc),
                        "timestamp": utc_now().isoformat(),
                    },
                )
                self.logger.warning(
                    "Failed arXiv PDF download",
                    extra={"context": {"arxiv_id": paper.arxiv_id, "error": str(exc)}},
                )
                return {"paper": paper, "status": "failed", "error": str(exc), "pdf_path": pdf_path, "metadata_path": metadata_path}

    async def ingest(self) -> dict[str, int]:
        candidates = await self.collect_candidates()
        if not candidates:
            return {"candidates": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        max_pdfs = self.config.arxiv.max_pdf_downloads_per_run
        selected = candidates[:max_pdfs]
        semaphore = asyncio.Semaphore(self.config.arxiv.pdf_download_concurrency)

        async with httpx.AsyncClient() as client:
            tasks = [self._download_pdf(client, semaphore, paper) for paper in selected]
            results = []
            for future in tqdm.as_completed(tasks, total=len(tasks), desc="Downloading arXiv PDFs"):
                results.append(await future)

        stats = {"candidates": len(candidates), "downloaded": 0, "skipped": 0, "failed": 0}
        saved_records: list[dict[str, Any]] = []
        last_updated_at: str | None = None

        for result in results:
            paper: ArxivPaperRecord = result["paper"]
            status = result["status"]
            if status == "failed":
                stats["failed"] += 1
                self.context.manifest_store.mark(
                    paper.arxiv_id,
                    source="arxiv",
                    status="failed",
                    payload={"error": result["error"], "pdf_url": paper.pdf_url},
                )
                continue

            if status == "skipped":
                stats["skipped"] += 1
            else:
                stats["downloaded"] += 1

            paper.pdf_path = Path(result["pdf_path"])
            paper.metadata_path = Path(result["metadata_path"])
            paper.download_timestamp = utc_now()
            self.context.metadata_store.save_arxiv_record(paper.metadata_path, paper)
            self.context.manifest_store.mark(
                paper.arxiv_id,
                source="arxiv",
                status=status,
                payload={
                    "title": paper.title,
                    "categories": paper.categories,
                    "updated_at": paper.updated_at.isoformat(),
                    "pdf_path": str(paper.pdf_path),
                    "metadata_path": str(paper.metadata_path),
                },
            )
            saved_records.append(paper.model_dump(mode="json"))
            if last_updated_at is None or paper.updated_at.isoformat() > last_updated_at:
                last_updated_at = paper.updated_at.isoformat()

        if saved_records and self.config.storage.parquet_export_enabled:
            summary_path = self.config.storage.metadata_dir / "arxiv_records_latest.parquet"
            self.context.metadata_store.export_records_to_parquet(summary_path, saved_records)

        if last_updated_at:
            self.context.state_store.set("arxiv.last_updated_at", last_updated_at)
            self.context.state_store.set("arxiv.last_run_at", utc_now().isoformat())

        self.logger.info("Completed arXiv ingestion", extra={"context": stats})
        return stats
