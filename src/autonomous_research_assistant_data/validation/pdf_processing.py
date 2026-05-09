"""Validation helpers for processed PDF artifacts."""

from __future__ import annotations

from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.storage.file_store import read_json


class PdfProcessingValidator:
    """Validate processed PDF artifacts and chunk outputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def validate(self) -> dict[str, int]:
        validation_files = list(self.config.pdf_processing.validation_dir.rglob("*.json"))
        total = 0
        needs_review = 0
        missing_chunks = 0
        missing_sections = 0
        missing_references = 0
        missing_repaired = 0
        missing_citations = 0
        missing_equation_blocks = 0
        missing_isolated_figures = 0
        missing_isolated_tables = 0
        missing_heading_analysis = 0
        missing_dedup_reports = 0
        low_quality_reports = 0
        legacy_reports = 0
        caption_leakage_reports = 0
        equation_corruption_reports = 0
        duplicated_section_reports = 0
        malformed_heading_reports = 0
        duplicate_paragraph_reports = 0
        column_merge_reports = 0
        failed = 0

        for validation_file in validation_files:
            total += 1
            payload = read_json(validation_file)
            status = payload.get("status")
            if status == "needs_review":
                needs_review += 1
            if status == "failed":
                failed += 1
            if int(payload.get("low_quality_chunk_count", 0)) > 0:
                low_quality_reports += 1
            legacy_mode = "reference_leakage_count" not in payload
            if legacy_mode:
                legacy_reports += 1
            if int(payload.get("caption_contamination_count", 0)) > 0:
                caption_leakage_reports += 1
            if int(payload.get("equation_corruption_count", 0)) > 0:
                equation_corruption_reports += 1
            if int(payload.get("duplicated_section_count", 0)) > 0:
                duplicated_section_reports += 1
            if int(payload.get("malformed_heading_count", 0)) > 0:
                malformed_heading_reports += 1
            if int(payload.get("duplicate_paragraph_count", 0)) > 0:
                duplicate_paragraph_reports += 1
            if int(payload.get("column_merge_corruption_count", 0)) > 0:
                column_merge_reports += 1

            source_pdf = Path(payload.get("source_pdf", ""))
            if source_pdf and not source_pdf.exists():
                missing_chunks += 1
            paper_id = payload.get("paper_id")
            if paper_id:
                chunk_matches = list(self.config.pdf_processing.chunks_dir.rglob(f"{paper_id}.json"))
                section_matches = list(self.config.pdf_processing.sections_dir.rglob(f"{paper_id}.json"))
                reference_matches = list(self.config.pdf_processing.references_dir.rglob(f"{paper_id}.json"))
                repaired_matches = list(self.config.pdf_processing.repaired_text_dir.rglob(f"{paper_id}.json"))
                citation_matches = list(self.config.pdf_processing.citations_dir.rglob(f"{paper_id}.json"))
                equation_matches = list(self.config.pdf_processing.equation_blocks_dir.rglob(f"{paper_id}.json"))
                figure_matches = list(self.config.pdf_processing.isolated_figures_dir.rglob(f"{paper_id}.json"))
                table_matches = list(self.config.pdf_processing.isolated_tables_dir.rglob(f"{paper_id}.json"))
                heading_matches = list(self.config.pdf_processing.heading_analysis_dir.rglob(f"{paper_id}.json"))
                dedup_matches = list(self.config.pdf_processing.dedup_reports_dir.rglob(f"{paper_id}.json"))
                if not chunk_matches:
                    missing_chunks += 1
                if not section_matches:
                    missing_sections += 1
                if not repaired_matches:
                    missing_repaired += 1
                if not citation_matches:
                    missing_citations += 1
                if not equation_matches:
                    missing_equation_blocks += 1
                if not figure_matches:
                    missing_isolated_figures += 1
                if not table_matches:
                    missing_isolated_tables += 1
                if not heading_matches:
                    missing_heading_analysis += 1
                if not dedup_matches:
                    missing_dedup_reports += 1
                if not legacy_mode and not reference_matches:
                    missing_references += 1
        return {
            "reports": total,
            "needs_review": needs_review,
            "failed": failed,
            "missing_sources": missing_chunks,
            "missing_sections": missing_sections,
            "missing_references": missing_references,
            "missing_repaired": missing_repaired,
            "missing_citations": missing_citations,
            "missing_equation_blocks": missing_equation_blocks,
            "missing_isolated_figures": missing_isolated_figures,
            "missing_isolated_tables": missing_isolated_tables,
            "missing_heading_analysis": missing_heading_analysis,
            "missing_dedup_reports": missing_dedup_reports,
            "low_quality_reports": low_quality_reports,
            "legacy_reports": legacy_reports,
            "caption_leakage_reports": caption_leakage_reports,
            "equation_corruption_reports": equation_corruption_reports,
            "duplicated_section_reports": duplicated_section_reports,
            "malformed_heading_reports": malformed_heading_reports,
            "duplicate_paragraph_reports": duplicate_paragraph_reports,
            "column_merge_reports": column_merge_reports,
        }
