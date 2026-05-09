"""Scientific content repair and corpus quality layer."""

from __future__ import annotations

import re
from collections import Counter

from autonomous_research_assistant_data.models.common import CleanParagraph, ExcludedArtifact, ExtractedDocument
from autonomous_research_assistant_data.processing.citations.parser import CitationParser
from autonomous_research_assistant_data.processing.dedup.paragraph_dedup import ParagraphDedupEngine
from autonomous_research_assistant_data.processing.equations.structure import EquationStructureEngine
from autonomous_research_assistant_data.processing.layout.multicolumn import MultiColumnLayoutEngine
from autonomous_research_assistant_data.processing.layout.region_detector import RegionIsolationEngine
from autonomous_research_assistant_data.processing.structure.heading_intelligence import HeadingIntelligenceEngine
from autonomous_research_assistant_data.processing.utils import semantic_density_score


class ScientificContentRepairEngine:
    """Repair scientific content while preserving downstream retrieval structure."""

    CAPTION_PATTERN = re.compile(r"^(figure|fig\.|table|algorithm|scheme)\s+\d+[A-Za-z]?(?:[\s:.\-]|$)", re.IGNORECASE)
    FIGURE_REF_FRAGMENT_PATTERN = re.compile(r"^(?:\(?[a-z]\)|left|right|top|bottom|ours|baseline|input|output)$", re.IGNORECASE)
    HEADER_FOOTER_PAGE_PATTERN = re.compile(r"^(page\s+)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)
    TABLE_LINE_PATTERN = re.compile(r"(\b\d+(?:\.\d+)?\b.*){3,}")
    OCR_ARTIFACT_PATTERN = re.compile(r"(cid:\d+|[A-Za-z]{1}\s){5,}|[^\w\s]{4,}")
    SENTENCE_END_PATTERN = re.compile(r"[.!?]\)?[\"']?$")
    SECTION_DUPLICATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+\1[\s.:]+", re.IGNORECASE)
    BOILERPLATE_PATTERN = re.compile(r"(preprint|under review|copyright|all rights reserved|conference on|arxiv:)", re.IGNORECASE)
    INLINE_FIGURE_PATTERN = re.compile(r"^\([a-z]\)\s+", re.IGNORECASE)

    def __init__(self, cleaning_rules: dict[str, bool]) -> None:
        self.cleaning_rules = cleaning_rules
        self.citation_parser = CitationParser()
        self.heading_engine = HeadingIntelligenceEngine()
        self.layout_engine = MultiColumnLayoutEngine()
        self.region_engine = RegionIsolationEngine()
        self.dedup_engine = ParagraphDedupEngine()

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\u00ad", "")
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _edge_line_frequencies(self, document: ExtractedDocument) -> Counter[str]:
        counter: Counter[str] = Counter()
        for page in document.pages:
            edge_lines = page.lines[:3] + page.lines[-3:]
            for line in edge_lines:
                normalized = self._normalize_text(line)
                if 2 <= len(normalized) <= 180:
                    counter[normalized] += 1
        return counter

    def _is_header_footer_artifact(self, text: str, repeated_lines: set[str]) -> bool:
        if text in repeated_lines:
            return True
        if self.HEADER_FOOTER_PAGE_PATTERN.fullmatch(text):
            return True
        return bool(self.BOILERPLATE_PATTERN.search(text)) and len(text) < 180

    def _punctuation_density(self, text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for char in text if not char.isalnum() and not char.isspace()) / max(len(text), 1)

    def _classify_noise(self, paragraph: CleanParagraph) -> list[str]:
        text = paragraph.text.strip()
        labels: list[str] = []
        if self.OCR_ARTIFACT_PATTERN.search(text):
            labels.append("ocr_artifact")
        if self.TABLE_LINE_PATTERN.search(text) or (sum(char.isdigit() for char in text) >= 8 and len(text.split()) <= 22):
            labels.append("table_bleed")
        if self.CAPTION_PATTERN.match(text):
            labels.append("caption_candidate")
        if self.INLINE_FIGURE_PATTERN.match(text):
            labels.append("inline_figure_marker")
        if self.FIGURE_REF_FRAGMENT_PATTERN.fullmatch(text):
            labels.append("diagram_label")
        if self.SECTION_DUPLICATION_PATTERN.match(text):
            labels.append("duplicated_section_heading")
        if paragraph.metadata.get("multi_column_page") and len(text.split()) > 70:
            labels.append("multi_column_risk")
        if len(text.split()) <= 14 and self._punctuation_density(text) > 0.18 and sum(char.isupper() for char in text) >= 2:
            labels.append("layout_contamination")
        return labels

    def _should_drop_paragraph(self, paragraph: CleanParagraph) -> tuple[bool, str | None]:
        labels = paragraph.noise_classifications
        if "caption_candidate" in labels and paragraph.structural_role not in {"section_heading"}:
            return True, "caption"
        if "diagram_label" in labels:
            return True, "diagram_label"
        if "table_bleed" in labels and paragraph.text.count(" ") < 18:
            return True, "table_bleed"
        return False, None

    def _merge_sentence_fragments(self, paragraphs: list[CleanParagraph]) -> tuple[list[CleanParagraph], int]:
        if not paragraphs:
            return [], 0
        merged: list[CleanParagraph] = [paragraphs[0].model_copy(deep=True)]
        merge_count = 0
        for current in paragraphs[1:]:
            last = merged[-1]
            if (
                last.structural_role not in {"equation_block", "section_heading", "list_item", "table_block"}
                and current.structural_role not in {"equation_block", "section_heading", "list_item", "table_block"}
                and not self.SENTENCE_END_PATTERN.search(last.text)
                and current.text[:1].islower()
                and last.page_end == current.page_number
            ):
                joiner = "" if last.text.endswith("-") else " "
                last.text = f"{last.text[:-1] if last.text.endswith('-') else last.text}{joiner}{current.text}".strip()
                last.page_end = current.page_end or current.page_number
                last.contains_citation = last.contains_citation or current.contains_citation
                last.noise_classifications = list(dict.fromkeys(last.noise_classifications + current.noise_classifications))
                last.repair_confidence = min(last.repair_confidence, current.repair_confidence)
                last.semantic_density_score = semantic_density_score(last.text)
                merge_count += 1
                continue
            merged.append(current.model_copy(deep=True))
        return merged, merge_count

    def _tag_structural_roles(self, paragraphs: list[CleanParagraph]) -> None:
        for paragraph in paragraphs:
            text = paragraph.text.strip()
            if paragraph.is_equation:
                paragraph.structural_role = "equation_block"
            elif paragraph.section_hint == "heading":
                paragraph.structural_role = "section_heading"
            elif re.match(r"^(\d+\.|\([a-z]\)|[-*])\s+", text):
                paragraph.structural_role = "list_item"
            elif self.CAPTION_PATTERN.match(text):
                paragraph.structural_role = "caption"
            elif self.TABLE_LINE_PATTERN.search(text):
                paragraph.structural_role = "table_block"
            elif re.match(r"^(theorem|lemma|proof|proposition|corollary)\b", text, flags=re.IGNORECASE):
                paragraph.structural_role = "theorem_block"

    def repair(
        self,
        paper_id: str,
        arxiv_id: str,
        document: ExtractedDocument,
        paragraphs: list[CleanParagraph],
        *,
        layout_aware: bool = True,
        dedup_strict: bool = False,
        equation_repair_level: str = "balanced",
        disable_column_reconstruction: bool = False,
    ) -> dict[str, object]:
        edge_frequencies = self._edge_line_frequencies(document)
        repeated_lines = {line for line, count in edge_frequencies.items() if count >= max(2, int(document.page_count * 0.35))}
        normalized: list[CleanParagraph] = []
        excluded_artifacts: list[ExcludedArtifact] = []
        suppression_stats = Counter()

        for paragraph in paragraphs:
            candidate = paragraph.model_copy(deep=True)
            candidate.text = self._normalize_text(candidate.text)
            if self._is_header_footer_artifact(candidate.text, repeated_lines):
                suppression_stats["header_footer"] += 1
                excluded_artifacts.append(
                    ExcludedArtifact(
                        artifact_id=f"{paper_id}-r{len(excluded_artifacts):04d}",
                        artifact_type="header_footer",
                        text=candidate.text,
                        page_number=candidate.page_number,
                        confidence=0.95,
                    )
                )
                continue
            normalized.append(candidate)

        normalized, layout_stats = self.layout_engine.reconstruct(document, normalized, enabled=layout_aware and not disable_column_reconstruction)
        normalized, heading_records, heading_stats = self.heading_engine.analyze(paper_id, arxiv_id, document, normalized)

        repaired_paragraphs: list[CleanParagraph] = []
        for paragraph in normalized:
            candidate = paragraph.model_copy(deep=True)
            candidate.noise_classifications = self._classify_noise(candidate)
            drop, artifact_type = self._should_drop_paragraph(candidate)
            if drop:
                suppression_stats[artifact_type or "unknown"] += 1
                excluded_artifacts.append(
                    ExcludedArtifact(
                        artifact_id=f"{paper_id}-r{len(excluded_artifacts):04d}",
                        artifact_type=artifact_type or "artifact",
                        text=candidate.text,
                        page_number=candidate.page_number,
                        confidence=0.88,
                    )
                )
                continue
            if "ocr_artifact" in candidate.noise_classifications:
                candidate.repair_confidence = min(candidate.repair_confidence, 0.7)
            if "table_bleed" in candidate.noise_classifications:
                candidate.repair_confidence = min(candidate.repair_confidence, 0.72)
            if "multi_column_risk" in candidate.noise_classifications:
                candidate.repair_confidence = min(candidate.repair_confidence, 0.78)
            candidate.semantic_density_score = semantic_density_score(candidate.text)
            repaired_paragraphs.append(candidate)

        self._tag_structural_roles(repaired_paragraphs)
        repaired_paragraphs, figure_regions, table_regions, region_stats = self.region_engine.isolate(paper_id, arxiv_id, repaired_paragraphs)
        equation_engine = EquationStructureEngine(repair_level=equation_repair_level)
        repaired_paragraphs, equation_blocks, equation_stats = equation_engine.repair(paper_id, arxiv_id, repaired_paragraphs)
        repaired_paragraphs, dedup_groups, dedup_stats = self.dedup_engine.deduplicate(paper_id, arxiv_id, repaired_paragraphs, strict=dedup_strict)
        self._tag_structural_roles(repaired_paragraphs)
        repaired_paragraphs, sentence_merge_count = self._merge_sentence_fragments(repaired_paragraphs)

        for paragraph in repaired_paragraphs:
            citation_data = self.citation_parser.parse(paragraph.text, prefix=paragraph.paragraph_id)
            paragraph.citation_spans = citation_data["citation_spans"]
            paragraph.contains_citation = bool(paragraph.citation_spans)
            paragraph.metadata["citation_density"] = citation_data["citation_density"]
            paragraph.metadata["citation_entities"] = citation_data["citation_entities"]
            paragraph.metadata["citation_offsets"] = citation_data["citation_offsets"]

        repair_confidence = round(sum(item.repair_confidence for item in repaired_paragraphs) / max(len(repaired_paragraphs), 1), 4)
        analytics = {
            "header_footer_candidates": len(repeated_lines),
            "suppression_counts": dict(suppression_stats),
            "heading_statistics": heading_stats,
            "layout_statistics": layout_stats,
            "equation_statistics": equation_stats,
            "dedup_statistics": dedup_stats,
            "region_statistics": region_stats,
            "sentence_merges": sentence_merge_count,
            "repair_confidence_mean": repair_confidence,
            "noise_histogram": dict(Counter(label for paragraph in repaired_paragraphs for label in paragraph.noise_classifications)),
        }
        report = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "repair_confidence": repair_confidence,
            "paragraph_count_before": len(paragraphs),
            "paragraph_count_after": len(repaired_paragraphs),
            "suppressed_artifact_count": len(excluded_artifacts),
            "suppressed_artifact_types": dict(Counter(item.artifact_type for item in excluded_artifacts)),
            "analytics": analytics,
        }
        return {
            "paragraphs": repaired_paragraphs,
            "equation_blocks": equation_blocks,
            "heading_records": heading_records,
            "figure_regions": figure_regions,
            "table_regions": table_regions,
            "dedup_groups": dedup_groups,
            "excluded_artifacts": excluded_artifacts,
            "report": report,
            "analytics": analytics,
        }
