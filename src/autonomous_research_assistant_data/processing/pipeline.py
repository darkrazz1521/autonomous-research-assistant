"""End-to-end PDF processing pipeline for elite scientific chunk generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from autonomous_research_assistant_data.chunking.semantic import SemanticChunker
from autonomous_research_assistant_data.models.common import CleanParagraph, ExtractedDocument, FrontMatterRecord, ProcessingReport
from autonomous_research_assistant_data.parsers.pdf_extractor import ScientificPdfExtractor
from autonomous_research_assistant_data.parsers.section_parser import ScientificSectionParser
from autonomous_research_assistant_data.processing.base import BaseProcessor, ProcessingContext
from autonomous_research_assistant_data.processing.repair.scientific_repair import ScientificContentRepairEngine
from autonomous_research_assistant_data.processing.text_cleaner import ScientificTextCleaner
from autonomous_research_assistant_data.processing.utils import safe_paper_id
from autonomous_research_assistant_data.storage.file_store import append_jsonl, ensure_directory, read_json
from autonomous_research_assistant_data.validators.chunks import ChunkValidator


class PdfProcessingPipeline(BaseProcessor):
    """Process arXiv PDFs into cleaned scientific sections and elite semantic chunks."""

    source_name = "pdf_processing"

    def __init__(
        self,
        context: ProcessingContext,
        *,
        force_reprocess: bool = False,
        repair_only: bool = False,
        skip_repair: bool = False,
        strict_validation: bool = False,
        layout_aware: bool | None = None,
        dedup_strict: bool | None = None,
        equation_repair_level: str | None = None,
        disable_column_reconstruction: bool = False,
    ) -> None:
        super().__init__(context)
        self.processing_config = self.config.pdf_processing
        self.force_reprocess = force_reprocess
        self.repair_only = repair_only
        self.skip_repair = skip_repair
        self.strict_validation = strict_validation
        self.layout_aware = self.processing_config.layout_aware_default if layout_aware is None else layout_aware
        self.dedup_strict = self.processing_config.dedup_strict_default if dedup_strict is None else dedup_strict
        self.equation_repair_level = equation_repair_level or self.processing_config.equation_repair_level
        self.disable_column_reconstruction = disable_column_reconstruction
        self.extractor = ScientificPdfExtractor(
            preferred_backend=self.processing_config.extraction_backend,
            fallback_backends=self.processing_config.fallback_backends,
        )
        self.cleaner = ScientificTextCleaner(
            cleaning_rules=self.processing_config.cleaning_rules,
            paragraph_merge_line_threshold=self.processing_config.paragraph_merge_line_threshold,
        )
        self.repair_engine = ScientificContentRepairEngine(self.processing_config.cleaning_rules)
        self.section_parser = ScientificSectionParser(self.processing_config.section_min_confidence)
        self.chunker = SemanticChunker(
            chunk_size=self.processing_config.chunk_size,
            chunk_overlap=self.processing_config.chunk_overlap,
            min_chunk_tokens=self.processing_config.min_chunk_tokens,
            max_chunk_tokens=self.processing_config.max_chunk_tokens,
            min_overlap_paragraphs=self.processing_config.min_overlap_paragraphs,
            max_overlap_paragraphs=self.processing_config.max_overlap_paragraphs,
            abstract_chunk_max_tokens=self.processing_config.abstract_chunk_max_tokens,
            repair_overlap_buffer_tokens=self.processing_config.repair_overlap_buffer_tokens,
        )
        validator_rules = dict(self.processing_config.validation_rules)
        validator_rules.setdefault("min_chunk_tokens", self.processing_config.min_chunk_tokens)
        validator_rules.setdefault("max_chunk_tokens", self.processing_config.max_chunk_tokens)
        self.validator = ChunkValidator(validator_rules)

    def _artifact_paths(self, arxiv_id: str, pdf_path: Path) -> dict[str, Path]:
        year = pdf_path.parent.parent.name
        month = pdf_path.parent.name
        paper_id = safe_paper_id(arxiv_id)
        return {
            "front_matter": self.processing_config.front_matter_dir / year / month / f"{paper_id}.json",
            "extracted": self.processing_config.extracted_text_dir / year / month / f"{paper_id}.json",
            "cleaned": self.processing_config.cleaned_text_dir / year / month / f"{paper_id}.json",
            "repaired": self.processing_config.repaired_text_dir / year / month / f"{paper_id}.json",
            "sections": self.processing_config.sections_dir / year / month / f"{paper_id}.json",
            "chunks": self.processing_config.chunks_dir / year / month / f"{paper_id}.json",
            "references": self.processing_config.references_dir / year / month / f"{paper_id}.json",
            "citations": self.processing_config.citations_dir / year / month / f"{paper_id}.json",
            "equation_blocks": self.processing_config.equation_blocks_dir / year / month / f"{paper_id}.json",
            "isolated_figures": self.processing_config.isolated_figures_dir / year / month / f"{paper_id}.json",
            "isolated_tables": self.processing_config.isolated_tables_dir / year / month / f"{paper_id}.json",
            "heading_analysis": self.processing_config.heading_analysis_dir / year / month / f"{paper_id}.json",
            "dedup_reports": self.processing_config.dedup_reports_dir / year / month / f"{paper_id}.json",
            "repair_report": self.processing_config.repair_reports_dir / year / month / f"{paper_id}.json",
            "validation": self.processing_config.validation_dir / year / month / f"{paper_id}.json",
        }

    def _already_processed(self, arxiv_id: str, paths: dict[str, Path]) -> bool:
        if self.processing_config.overwrite_existing or self.force_reprocess:
            return False
        manifest_hit = self.context.manifest_store.exists(arxiv_id)
        if self.repair_only:
            required = ("repaired", "sections", "chunks", "references", "citations", "equation_blocks", "isolated_figures", "isolated_tables", "heading_analysis", "dedup_reports", "repair_report", "validation")
        else:
            required = tuple(paths.keys())
        return manifest_hit and all(paths[name].exists() for name in required)

    def _load_source_items(self) -> list[tuple[Path, Path, dict[str, Any]]]:
        pdf_files = sorted(self.processing_config.source_pdf_dir.rglob("*.pdf"))
        items: list[tuple[Path, Path, dict[str, Any]]] = []
        for pdf_path in pdf_files:
            metadata_path = self.processing_config.source_metadata_dir / pdf_path.relative_to(self.processing_config.source_pdf_dir)
            metadata_path = metadata_path.with_suffix(".json")
            if metadata_path.exists():
                items.append((pdf_path, metadata_path, read_json(metadata_path)))
        return items[: self.processing_config.max_papers_per_run]

    def _load_repair_inputs(self, paths: dict[str, Path]) -> tuple[ExtractedDocument, FrontMatterRecord, list[CleanParagraph], dict[str, Any]]:
        extracted_payload = read_json(paths["extracted"])
        cleaned_payload = read_json(paths["cleaned"])
        front_matter_payload = read_json(paths["front_matter"])
        if not extracted_payload or not cleaned_payload:
            raise FileNotFoundError("Repair-only mode requires existing extracted_text and cleaned_text artifacts.")
        return (
            ExtractedDocument.model_validate(extracted_payload),
            FrontMatterRecord.model_validate(front_matter_payload),
            [CleanParagraph.model_validate(item) for item in cleaned_payload.get("paragraphs", [])],
            cleaned_payload,
        )

    def _save_report(self, report_path: Path, report: ProcessingReport) -> None:
        self.context.metadata_store.save_record(report_path, report.model_dump(mode="json"))

    def _save_analytics(self, reports: list[ProcessingReport], all_chunks: list[dict[str, Any]], repair_reports: list[dict[str, Any]]) -> None:
        if not reports:
            return
        analytics_dir = ensure_directory(self.processing_config.analytics_dir)
        self.context.metadata_store.export_records_to_parquet(
            self.processing_config.reports_dir / "processing_reports_latest.parquet",
            [report.model_dump(mode="json") for report in reports],
        )
        if repair_reports:
            pd.DataFrame(repair_reports).to_parquet(analytics_dir / "repair_reports_latest.parquet", index=False)
        if not all_chunks:
            return

        chunk_frame = pd.DataFrame(all_chunks)
        chunk_frame.to_parquet(analytics_dir / "chunk_corpus_latest.parquet", index=False)
        token_buckets = pd.cut(chunk_frame["token_count_estimate"], bins=[0, 128, 256, 384, 512, 768, 2048], include_lowest=True).value_counts().sort_index().astype(int).to_dict()
        quality_bands = pd.cut(chunk_frame["retrieval_quality_score"], bins=[0, 0.4, 0.55, 0.7, 0.85, 1.0], include_lowest=True).value_counts().sort_index().astype(int).to_dict()
        section_quality = chunk_frame.groupby("section_name")["retrieval_quality_score"].mean().round(4).to_dict()
        equation_integrity = {
            "equation_chunk_count": int(chunk_frame["contains_equation"].sum()),
            "equation_density_mean": float(chunk_frame["equation_density"].mean()),
        }
        continuity = {
            "transition_quality_mean": float(chunk_frame["transition_quality_score"].mean()),
            "semantic_boundary_mean": float(chunk_frame["semantic_boundary_score"].mean()),
            "narrative_continuity_mean": float(chunk_frame["narrative_continuity_score"].mean()),
        }
        noise_series = chunk_frame["noise_classifications"].explode().dropna()
        self.context.metadata_store.save_record(
            analytics_dir / "chunk_statistics_latest.json",
            {
                "token_distribution": {
                    "count": int(chunk_frame["token_count_estimate"].count()),
                    "mean": float(chunk_frame["token_count_estimate"].mean()),
                    "min": int(chunk_frame["token_count_estimate"].min()),
                    "max": int(chunk_frame["token_count_estimate"].max()),
                },
                "token_histogram": {str(key): value for key, value in token_buckets.items()},
                "section_distribution": chunk_frame["section_name"].value_counts().to_dict(),
                "section_quality_histograms": section_quality,
                "quality_statistics": {
                    "retrieval_quality_mean": float(chunk_frame["retrieval_quality_score"].mean()),
                    "noise_score_mean": float(chunk_frame["noise_score"].mean()),
                    "coherence_score_mean": float(chunk_frame["coherence_score"].mean()),
                    "repair_confidence_mean": float(chunk_frame["repair_confidence"].mean()),
                    "structural_integrity_mean": float(chunk_frame["structural_integrity_score"].mean()),
                    "structural_anomaly_mean": float(chunk_frame["structural_anomaly_score"].mean()),
                },
                "quality_histogram": {str(key): value for key, value in quality_bands.items()},
                "equation_integrity_distributions": equation_integrity,
                "citation_statistics": {
                    "citation_chunk_count": int(chunk_frame["contains_citation"].sum()),
                    "citation_density_mean": float(chunk_frame["citation_density"].mean()),
                },
                "chunk_continuity_metrics": continuity,
                "layout_corruption_metrics": chunk_frame["corruption_categories"].explode().dropna().value_counts().to_dict() if "corruption_categories" in chunk_frame else {},
                "heading_confidence_analytics": [report.get("analytics", {}).get("heading_statistics", {}) for report in repair_reports],
                "noise_reduction_summary": noise_series.value_counts().to_dict() if not noise_series.empty else {},
            },
        )

    def process(self) -> dict[str, int]:
        items = self._load_source_items()
        stats = {"candidates": len(items), "processed": 0, "skipped": 0, "failed": 0, "chunks": 0}
        reports: list[ProcessingReport] = []
        all_chunk_records: list[dict[str, Any]] = []
        repair_reports: list[dict[str, Any]] = []
        failure_log = self.config.logging.failure_log_file

        for pdf_path, metadata_path, metadata in tqdm(items, desc="Processing PDFs"):
            arxiv_id = metadata.get("arxiv_id") or pdf_path.stem
            paper_id = safe_paper_id(arxiv_id)
            paths = self._artifact_paths(arxiv_id, pdf_path)
            if self._already_processed(arxiv_id, paths):
                stats["skipped"] += 1
                continue
            try:
                for path in paths.values():
                    ensure_directory(path.parent)

                if self.repair_only:
                    extracted, front_matter, paragraphs, cleaned_payload = self._load_repair_inputs(paths)
                    cleaning_payload = {
                        "stats": cleaned_payload.get("cleaning_stats", {}),
                        "excluded_artifacts": cleaned_payload.get("excluded_artifacts", []),
                        "references": read_json(paths["references"]).get("references", []),
                    }
                    cleaned_text = cleaned_payload.get("cleaned_text", "")
                else:
                    extracted = self.extractor.extract(paper_id=paper_id, arxiv_id=arxiv_id, pdf_path=pdf_path, abstract=metadata.get("abstract"))
                    cleaned_text, paragraphs, cleaning_payload = self.cleaner.clean(extracted)
                    front_matter = FrontMatterRecord.model_validate(cleaning_payload["front_matter"])

                if self.skip_repair:
                    repair_payload = {
                        "paragraphs": paragraphs,
                        "equation_blocks": [],
                        "heading_records": [],
                        "figure_regions": [],
                        "table_regions": [],
                        "dedup_groups": [],
                        "excluded_artifacts": [],
                        "report": {"paper_id": paper_id, "arxiv_id": arxiv_id, "repair_confidence": 1.0, "paragraph_count_before": len(paragraphs), "paragraph_count_after": len(paragraphs), "suppressed_artifact_count": 0, "suppressed_artifact_types": {}, "analytics": {}},
                    }
                else:
                    repair_payload = self.repair_engine.repair(
                        paper_id,
                        arxiv_id,
                        extracted,
                        paragraphs,
                        layout_aware=self.layout_aware,
                        dedup_strict=self.dedup_strict,
                        equation_repair_level=self.equation_repair_level,
                        disable_column_reconstruction=self.disable_column_reconstruction,
                    )

                repaired_paragraphs = repair_payload["paragraphs"]
                sections = self.section_parser.parse(paper_id=paper_id, arxiv_id=arxiv_id, paragraphs=repaired_paragraphs)
                chunks = self.chunker.chunk_sections(paper_id=paper_id, arxiv_id=arxiv_id, source_pdf=pdf_path, sections=sections, front_matter=front_matter)

                all_citation_spans = []
                citation_entities: set[str] = set()
                for paragraph in repaired_paragraphs:
                    all_citation_spans.extend(span.model_dump(mode="json") for span in paragraph.citation_spans)
                    citation_entities.update(paragraph.metadata.get("citation_entities", []))

                report = self.validator.validate(
                    paper_id=paper_id,
                    arxiv_id=arxiv_id,
                    source_pdf=pdf_path,
                    extraction_backend=extracted.extraction_backend,
                    extraction_quality_score=extracted.extraction_quality_score,
                    section_count=len(sections),
                    sections=sections,
                    chunks=chunks,
                    repair_report=repair_payload["report"],
                )
                if self.strict_validation and report.status != "ready":
                    report.status = "failed"
                report.extra = {
                    "source_metadata_path": str(metadata_path),
                    "paragraph_count": len(repaired_paragraphs),
                    "cleaning_stats": cleaning_payload["stats"],
                    "page_count": extracted.page_count,
                    "excluded_artifact_count": len(cleaning_payload["excluded_artifacts"]),
                    "reference_count": len(cleaning_payload["references"]),
                    "repair_report_path": str(paths["repair_report"]),
                }

                if not self.repair_only:
                    self.context.metadata_store.save_record(paths["front_matter"], front_matter.model_dump(mode="json"))
                    self.context.metadata_store.save_record(paths["extracted"], extracted.model_dump(mode="json"))
                    self.context.metadata_store.save_record(paths["cleaned"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "cleaned_text": cleaned_text, "paragraphs": [item.model_dump(mode="json") for item in paragraphs], "excluded_artifacts": cleaning_payload["excluded_artifacts"], "cleaning_stats": cleaning_payload["stats"]})

                self.context.metadata_store.save_record(paths["repaired"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "repaired_text": "\n\n".join(item.text for item in repaired_paragraphs), "paragraphs": [item.model_dump(mode="json") for item in repaired_paragraphs], "excluded_artifacts": [item.model_dump(mode="json") for item in repair_payload["excluded_artifacts"]], "repair_report": repair_payload["report"]})
                self.context.metadata_store.save_record(paths["sections"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "sections": [item.model_dump(mode="json") for item in sections]})
                self.context.metadata_store.save_record(paths["chunks"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "chunks": [item.model_dump(mode="json") for item in chunks]})
                self.context.metadata_store.save_record(paths["references"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "references": cleaning_payload["references"]})
                self.context.metadata_store.save_record(paths["citations"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "citation_spans": all_citation_spans, "citation_entities": sorted(citation_entities), "citation_density": round(len(all_citation_spans) / max(sum(chunk.token_count_estimate for chunk in chunks), 1), 4)})
                self.context.metadata_store.save_record(paths["equation_blocks"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "equation_blocks": [item.model_dump(mode="json") for item in repair_payload["equation_blocks"]]})
                self.context.metadata_store.save_record(paths["isolated_figures"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "isolated_figures": [item.model_dump(mode="json") for item in repair_payload.get("figure_regions", [])]})
                self.context.metadata_store.save_record(paths["isolated_tables"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "isolated_tables": [item.model_dump(mode="json") for item in repair_payload.get("table_regions", [])]})
                self.context.metadata_store.save_record(paths["heading_analysis"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "headings": [item.model_dump(mode="json") for item in repair_payload.get("heading_records", [])]})
                self.context.metadata_store.save_record(paths["dedup_reports"], {"paper_id": paper_id, "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "dedup_groups": [item.model_dump(mode="json") for item in repair_payload.get("dedup_groups", [])]})
                self.context.metadata_store.save_record(paths["repair_report"], repair_payload["report"])
                self._save_report(paths["validation"], report)

                self.context.manifest_store.mark(
                    arxiv_id,
                    source="pdf_processing",
                    status=report.status,
                    payload={
                        "paper_id": paper_id,
                        "source_pdf": str(pdf_path),
                        "metadata_path": str(metadata_path),
                        "chunk_count": len(chunks),
                        "section_count": len(sections),
                        "extraction_backend": extracted.extraction_backend,
                        "validation_path": str(paths["validation"]),
                        "chunks_path": str(paths["chunks"]),
                        "references_path": str(paths["references"]),
                        "front_matter_path": str(paths["front_matter"]),
                        "repaired_text_path": str(paths["repaired"]),
                        "citations_path": str(paths["citations"]),
                        "equation_blocks_path": str(paths["equation_blocks"]),
                        "isolated_figures_path": str(paths["isolated_figures"]),
                        "isolated_tables_path": str(paths["isolated_tables"]),
                        "heading_analysis_path": str(paths["heading_analysis"]),
                        "dedup_reports_path": str(paths["dedup_reports"]),
                        "repair_report_path": str(paths["repair_report"]),
                    },
                )
                self.context.state_store.set("pdf_processing.last_processed_arxiv_id", arxiv_id)
                stats["processed"] += 1
                stats["chunks"] += len(chunks)
                reports.append(report)
                repair_reports.append(repair_payload["report"])
                all_chunk_records.extend(chunk.model_dump(mode="json") for chunk in chunks)
            except Exception as exc:
                append_jsonl(failure_log, {"source": "pdf_processing", "arxiv_id": arxiv_id, "source_pdf": str(pdf_path), "error": str(exc)})
                self.context.manifest_store.mark(arxiv_id, source="pdf_processing", status="failed", payload={"paper_id": paper_id, "source_pdf": str(pdf_path), "error": str(exc)})
                self.logger.warning("Failed PDF processing item", extra={"context": {"arxiv_id": arxiv_id, "pdf_path": str(pdf_path), "error": str(exc)}})
                stats["failed"] += 1

        self.context.state_store.set("pdf_processing.last_run_at", reports[-1].processed_at.isoformat() if reports else None)
        self._save_analytics(reports, all_chunk_records, repair_reports)
        self.logger.info("Completed PDF processing", extra={"context": stats})
        return stats
