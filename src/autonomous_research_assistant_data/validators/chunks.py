"""Advanced chunk validation and retrieval-quality analytics."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ChunkRecord, ProcessingReport, SectionRecord


class ChunkValidator:
    """Validate chunk quality and generate corpus-quality metrics."""

    CAPTION_PATTERN = re.compile(r"^(figure|fig\.|table|algorithm)\s+\d+", re.IGNORECASE)

    def __init__(self, validation_rules: dict[str, float | int]) -> None:
        self.validation_rules = validation_rules

    def _noise_ratio(self, text: str) -> float:
        if not text:
            return 1.0
        return len(re.findall(r"[^\w\s\.,;:\-\(\)\[\]/%\\=+*<>≤≥≈ΣΠ∂∇θλβγ]", text)) / max(len(text), 1)

    def _reference_leakage(self, text: str) -> bool:
        return bool(re.search(r"\bdoi:|\bpp\.\s*\d+|\bProceedings of\b|\bIn:\s+[A-Z]", text))

    def _equation_fragmented(self, text: str) -> bool:
        open_parens = text.count("(") + text.count("[") + text.count("{")
        close_parens = text.count(")") + text.count("]") + text.count("}")
        return abs(open_parens - close_parens) >= 3 or bool(re.search(r"(=|\\sum|\\frac|≤|≥|≈).{0,8}$", text))

    def _ocr_corrupted(self, text: str) -> bool:
        return bool(re.search(r"(cid:\d+|(?:\b[A-Za-z]\b\s+){5,}|�)", text))

    def _unicode_corrupted(self, text: str) -> bool:
        return bool(re.search(r"(â‰|âˆ|Ã—|ï¬)", text))

    def validate(
        self,
        paper_id: str,
        arxiv_id: str,
        source_pdf,
        extraction_backend: str,
        extraction_quality_score: float,
        section_count: int,
        sections: list[SectionRecord],
        chunks: list[ChunkRecord],
        repair_report: dict[str, object] | None = None,
    ) -> ProcessingReport:
        seen_hashes: set[str] = set()
        seen_section_names: set[str] = set()
        duplicate_chunk_count = tiny_chunk_count = oversized_chunk_count = empty_chunk_count = 0
        equation_fragmentation_count = reference_leakage_count = incoherent_chunk_count = low_quality_chunk_count = 0
        ocr_corruption_count = equation_corruption_count = table_bleed_count = caption_contamination_count = 0
        duplicated_section_count = malformed_unicode_count = incomplete_reference_count = chunk_fragmentation_count = 0
        malformed_heading_count = duplicate_paragraph_count = column_merge_corruption_count = 0
        layout_contamination_count = figure_leakage_count = chunk_discontinuity_count = 0
        max_noise_ratio = 0.0
        total_overlap = 0
        min_chunk_chars = int(self.validation_rules.get("min_chunk_chars", 250))
        min_retrieval_quality_score = float(self.validation_rules.get("min_retrieval_quality_score", 0.5))
        previous_paragraph_ids: set[str] = set()
        previous_section_name = None

        for section in sections:
            lowered = section.section_name.lower()
            if lowered in seen_section_names and section.canonical_section_label not in {"results", "discussion", "appendix"}:
                duplicated_section_count += 1
            if len(section.section_name.split()) > 10:
                malformed_heading_count += 1
            seen_section_names.add(lowered)

        for chunk in chunks:
            text = chunk.chunk_text.strip()
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if not text:
                empty_chunk_count += 1
            if len(text) < min_chunk_chars or chunk.token_count_estimate < int(self.validation_rules.get("min_chunk_tokens", 80)):
                tiny_chunk_count += 1
            if chunk.token_count_estimate > int(self.validation_rules.get("max_chunk_tokens", 900)):
                oversized_chunk_count += 1
            if chunk.semantic_hash in seen_hashes:
                duplicate_chunk_count += 1
            if chunk.semantic_hash:
                seen_hashes.add(chunk.semantic_hash)

            noise_ratio = self._noise_ratio(text)
            max_noise_ratio = max(max_noise_ratio, noise_ratio)
            if chunk.contains_equation and chunk.equation_density >= 0.25 and self._equation_fragmented(text):
                equation_fragmentation_count += 1
                equation_corruption_count += 1
            if self._reference_leakage(text):
                reference_leakage_count += 1
            if chunk.coherence_score < 0.4:
                incoherent_chunk_count += 1
            if chunk.retrieval_quality_score < min_retrieval_quality_score:
                low_quality_chunk_count += 1
            if self._ocr_corrupted(text):
                ocr_corruption_count += 1
            if self._unicode_corrupted(text):
                malformed_unicode_count += 1
            if "table_bleed" in chunk.noise_classifications:
                table_bleed_count += 1
            if "layout_contamination" in chunk.noise_classifications:
                layout_contamination_count += 1
            if "multi_column_risk" in chunk.noise_classifications:
                column_merge_corruption_count += 1
            if self.CAPTION_PATTERN.search(first_line) and len(first_line.split()) <= 12:
                caption_contamination_count += 1
                figure_leakage_count += 1
            if chunk.contains_citation and text.endswith(("et al.,", "et al.", "[", "(")):
                incomplete_reference_count += 1
            if previous_section_name == chunk.section_name and chunk.chunk_text[:1].islower():
                chunk_fragmentation_count += 1
            previous_section_name = chunk.section_name
            if chunk.transition_quality_score < 0.52 or chunk.narrative_continuity_score < 0.52:
                chunk_discontinuity_count += 1

            current_ids = set(chunk.paragraph_ids)
            if previous_paragraph_ids:
                overlap = len(previous_paragraph_ids.intersection(current_ids))
                total_overlap += overlap
                if overlap >= max(4, int(len(current_ids) * 0.8)):
                    duplicate_paragraph_count += overlap
            previous_paragraph_ids = current_ids

        overlap_duplicate_ratio = total_overlap / max(sum(len(chunk.paragraph_ids) for chunk in chunks), 1)
        if overlap_duplicate_ratio > float(self.validation_rules.get("max_duplicate_chunk_ratio", 0.2)):
            chunk_fragmentation_count += 1

        repair_confidence = float((repair_report or {}).get("repair_confidence", 1.0))
        suppressed_count = int((repair_report or {}).get("suppressed_artifact_count", 0))
        repair_effectiveness = max(0.0, min(1.0, repair_confidence * 0.7 + min(suppressed_count / 20, 0.3)))
        status = "ready"
        if (
            empty_chunk_count > 0
            or max_noise_ratio > float(self.validation_rules.get("max_noise_ratio", 0.35))
            or reference_leakage_count > int(self.validation_rules.get("max_reference_leakage_count", 1))
            or equation_fragmentation_count > int(self.validation_rules.get("max_equation_fragmentation_count", 1))
            or caption_contamination_count > 0
            or malformed_unicode_count > 0
            or malformed_heading_count > 3
            or duplicate_paragraph_count > 4
            or column_merge_corruption_count > 0
            or chunk_discontinuity_count > max(4, len(chunks) // 3)
        ):
            status = "needs_review"

        return ProcessingReport(
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            source_pdf=source_pdf,
            extraction_backend=extraction_backend,
            extraction_quality_score=extraction_quality_score,
            section_count=section_count,
            chunk_count=len(chunks),
            empty_chunk_count=empty_chunk_count,
            duplicate_chunk_count=duplicate_chunk_count,
            tiny_chunk_count=tiny_chunk_count,
            oversized_chunk_count=oversized_chunk_count,
            overlap_duplicate_ratio=round(overlap_duplicate_ratio, 4),
            equation_fragmentation_count=equation_fragmentation_count,
            reference_leakage_count=reference_leakage_count,
            incoherent_chunk_count=incoherent_chunk_count,
            low_quality_chunk_count=low_quality_chunk_count,
            noise_ratio_max=round(max_noise_ratio, 4),
            ocr_corruption_count=ocr_corruption_count,
            equation_corruption_count=equation_corruption_count,
            table_bleed_count=table_bleed_count,
            caption_contamination_count=caption_contamination_count,
            duplicated_section_count=duplicated_section_count,
            malformed_unicode_count=malformed_unicode_count,
            incomplete_reference_count=incomplete_reference_count,
            chunk_fragmentation_count=chunk_fragmentation_count,
            repair_effectiveness_score=round(repair_effectiveness, 4),
            malformed_heading_count=malformed_heading_count,
            duplicate_paragraph_count=duplicate_paragraph_count,
            column_merge_corruption_count=column_merge_corruption_count,
            layout_contamination_count=layout_contamination_count,
            figure_leakage_count=figure_leakage_count,
            chunk_discontinuity_count=chunk_discontinuity_count,
            status=status,
            processed_at=chunks[0].processing_timestamp if chunks else utc_now(),
            extra={},
        )
