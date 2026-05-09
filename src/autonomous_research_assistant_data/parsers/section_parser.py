"""Hierarchical scientific section detection with canonical labels."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import CleanParagraph, SectionRecord
from autonomous_research_assistant_data.processing.sections.classifier import ScientificSectionClassifier


class ScientificSectionParser:
    """Detect sections using heading intelligence plus semantic classification."""

    INLINE_HEADING_PATTERNS = (
        re.compile(r"^(Related Work)\s+(?=[A-Z])"),
        re.compile(r"^(Conclusion|Conclusions)\s+(?=[A-Z])"),
        re.compile(r"^(Discussion)\s+(?=[A-Z])"),
        re.compile(r"^(Limitations?)\s+(?=[A-Z])"),
        re.compile(r"^(Introduction)\s+(?=[A-Z])"),
        re.compile(r"^(Abstract)\s+(?=[A-Z])"),
    )

    def __init__(self, min_confidence: float) -> None:
        self.min_confidence = min_confidence
        self.classifier = ScientificSectionClassifier()

    def _canonical_label(self, paragraph: CleanParagraph) -> tuple[str, float, str]:
        if paragraph.normalized_heading:
            inferred, score = self.classifier.classify(paragraph.normalized_heading)
            if inferred:
                return inferred, max(paragraph.heading_confidence, score), paragraph.normalized_heading
        inferred, score = self.classifier.classify(paragraph.text)
        return inferred or "body", score, paragraph.normalized_heading or paragraph.text.strip()

    def _split_inline_heading(self, paragraph: CleanParagraph) -> tuple[str, str] | None:
        if paragraph.section_hint == "heading":
            return None
        text = paragraph.text.strip()
        if len(text.split()) < 6:
            return None
        for pattern in self.INLINE_HEADING_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue
            heading = match.group(1).strip()
            body = text[match.end():].strip(" .:")
            if len(body) < 32:
                return None
            return heading, body
        return None

    def parse(self, paper_id: str, arxiv_id: str, paragraphs: list[CleanParagraph]) -> list[SectionRecord]:
        if not paragraphs:
            return []

        sections: list[SectionRecord] = []
        current_heading = "Front Matter"
        current_label = "front_matter"
        current_confidence = 0.4
        current_level = 1
        current_heading_raw = None
        current_paragraphs: list[CleanParagraph] = []
        section_index = 0

        def flush() -> None:
            nonlocal current_paragraphs, section_index
            if not current_paragraphs:
                return
            sections.append(
                SectionRecord(
                    section_id=f"{paper_id}-s{section_index:03d}",
                    paper_id=paper_id,
                    arxiv_id=arxiv_id,
                    section_index=section_index,
                    section_name=current_heading,
                    normalized_section_name=current_heading.lower(),
                    canonical_section_label=current_label,
                    heading_raw=current_heading_raw,
                    confidence=current_confidence,
                    level=current_level,
                    page_start=current_paragraphs[0].page_number,
                    page_end=current_paragraphs[-1].page_end or current_paragraphs[-1].page_number,
                    contains_references=current_label == "references",
                    paragraphs=current_paragraphs,
                )
            )
            section_index += 1
            current_paragraphs = []

        for paragraph in paragraphs:
            is_heading = paragraph.section_hint == "heading" and paragraph.heading_confidence >= self.min_confidence
            if is_heading:
                flush()
                label, confidence, display = self._canonical_label(paragraph)
                current_heading = display.strip().title() if display.islower() else display.strip()
                current_heading_raw = paragraph.text
                current_label = label
                current_confidence = max(confidence, paragraph.heading_confidence)
                current_level = paragraph.hierarchy_depth or 1
                continue
            inline_heading = self._split_inline_heading(paragraph)
            if inline_heading:
                flush()
                heading_text, body_text = inline_heading
                current_heading = heading_text
                current_heading_raw = heading_text
                current_label, current_confidence = self.classifier.classify(heading_text)
                current_label = current_label or "body"
                current_level = 1
                paragraph = paragraph.model_copy(deep=True)
                paragraph.text = body_text
                paragraph.semantic_density_score = paragraph.semantic_density_score
            if current_label == "references" and paragraph.contains_citation:
                paragraph.section_hint = "reference_entry"
            current_paragraphs.append(paragraph)

        flush()

        if not sections:
            sections.append(
                SectionRecord(
                    section_id=f"{paper_id}-s000",
                    paper_id=paper_id,
                    arxiv_id=arxiv_id,
                    section_index=0,
                    section_name="Body",
                    normalized_section_name="body",
                    canonical_section_label="body",
                    confidence=0.4,
                    paragraphs=paragraphs,
                )
            )
        return sections
