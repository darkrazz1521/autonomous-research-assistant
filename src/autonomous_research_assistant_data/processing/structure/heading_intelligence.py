"""Advanced heading detection for scientific PDFs."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import CleanParagraph, ExtractedDocument, HeadingRecord
from autonomous_research_assistant_data.processing.sections.classifier import ScientificSectionClassifier


class HeadingIntelligenceEngine:
    """Detect and normalize structural headings using layout and semantic heuristics."""

    NUMBERING_PATTERN = re.compile(r"^(?:(\d+(?:\.\d+)*)|([IVXLC]+)|Appendix\s+([A-Z]))[\s\.:]+", re.IGNORECASE)
    DUPLICATE_PREFIX_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+\1[\s\.:]+", re.IGNORECASE)
    HEADING_CASE_PATTERN = re.compile(r"^[A-Z0-9][A-Za-z0-9\s\-\&:/,()]{1,90}$")
    INLINE_FIGURE_PATTERN = re.compile(r"^\([a-z]\)\s+", re.IGNORECASE)
    BODY_VERB_PATTERN = re.compile(r"\b(is|are|was|were|shows|demonstrates|provide|proves|achieves|using|including|because)\b", re.IGNORECASE)

    def __init__(self) -> None:
        self.section_classifier = ScientificSectionClassifier()

    def _normalize_heading(self, text: str) -> str:
        text = self.DUPLICATE_PREFIX_PATTERN.sub("", text.strip())
        text = self.NUMBERING_PATTERN.sub("", text).strip(" .:")
        return re.sub(r"\s+", " ", text)

    def _hierarchy_depth(self, text: str) -> int:
        match = self.NUMBERING_PATTERN.match(text.strip())
        if not match:
            return 1
        if match.group(1):
            return len(match.group(1).split("."))
        return 1

    def _heading_type(self, text: str) -> str:
        lowered = text.lower()
        if lowered.startswith("appendix") or "supplement" in lowered:
            return "appendix"
        if self.NUMBERING_PATTERN.match(text):
            return "numbered"
        if text.isupper():
            return "display"
        return "semantic"

    def _titlecase_ratio(self, text: str) -> float:
        words = [word for word in re.split(r"\s+", text.strip()) if word]
        if not words:
            return 0.0
        starters = sum(1 for word in words if word[:1].isupper())
        return starters / len(words)

    def _layout_hint(self, document: ExtractedDocument, page_number: int, text: str) -> float:
        page = next((page for page in document.pages if page.page_number == page_number), None)
        if page is None:
            return 0.0
        try:
            index = page.lines.index(text)
        except ValueError:
            return 0.0
        top_boost = 0.08 if index <= 3 else 0.0
        isolation_boost = 0.06 if len(text.split()) <= 8 else 0.0
        return top_boost + isolation_boost

    def _confidence(self, document: ExtractedDocument, paragraph: CleanParagraph) -> tuple[float, str | None, str]:
        text = paragraph.text.strip()
        normalized = self._normalize_heading(text)
        canonical, semantic_score = self.section_classifier.classify(normalized)
        short_line = len(normalized.split()) <= 8
        case_score = 0.14 if self.HEADING_CASE_PATTERN.match(normalized) else 0.0
        numbering_score = 0.28 if self.NUMBERING_PATTERN.match(text) else 0.0
        short_score = 0.2 if short_line else -0.28
        punctuation_penalty = -0.24 if normalized.endswith(".") or normalized.count(",") >= 2 else 0.0
        inline_penalty = -0.45 if self.INLINE_FIGURE_PATTERN.match(normalized) else 0.0
        density_penalty = -0.35 if len(normalized.split()) > 10 else 0.0
        verb_penalty = -0.3 if self.BODY_VERB_PATTERN.search(normalized) and semantic_score < 0.88 else 0.0
        layout_score = self._layout_hint(document, paragraph.page_number, text)
        confidence = max(0.0, min(1.0, semantic_score + case_score + numbering_score + short_score + punctuation_penalty + inline_penalty + density_penalty + verb_penalty + layout_score))
        return confidence, canonical, normalized

    def analyze(self, paper_id: str, arxiv_id: str, document: ExtractedDocument, paragraphs: list[CleanParagraph]) -> tuple[list[CleanParagraph], list[HeadingRecord], dict[str, object]]:
        analyzed: list[CleanParagraph] = []
        headings: list[HeadingRecord] = []
        malformed = 0

        for paragraph in paragraphs:
            candidate = paragraph.model_copy(deep=True)
            confidence, canonical, normalized = self._confidence(document, candidate)
            candidate.heading_confidence = confidence
            candidate.normalized_heading = normalized if confidence >= 0.5 else None
            candidate.hierarchy_depth = self._hierarchy_depth(candidate.text) if confidence >= 0.5 else 0
            candidate.heading_type = self._heading_type(candidate.text) if confidence >= 0.5 else None
            if candidate.section_hint == "heading":
                candidate.section_hint = None
            if candidate.structural_role == "section_heading":
                candidate.structural_role = None
            numbered = self.NUMBERING_PATTERN.match(candidate.text) is not None
            semantic_title_like = normalized[:1].isupper() and self._titlecase_ratio(normalized) >= 0.5
            ends_like_body = candidate.text.rstrip().endswith((".", ";")) or not re.search(r"[A-Za-z]", normalized)
            eligible_heading = (
                confidence >= 0.76
                and len(normalized.split()) <= (8 if numbered else 6)
                and not ends_like_body
                and ((numbered and self._titlecase_ratio(normalized) >= 0.35) or (canonical is not None and semantic_title_like))
            )
            if eligible_heading:
                candidate.section_hint = "heading"
                candidate.structural_role = "section_heading"
                headings.append(
                    HeadingRecord(
                        heading_id=f"{paper_id}-h{len(headings):04d}",
                        paper_id=paper_id,
                        arxiv_id=arxiv_id,
                        text=candidate.text,
                        normalized_heading=normalized,
                        page_number=candidate.page_number,
                        paragraph_id=candidate.paragraph_id,
                        heading_confidence=confidence,
                        heading_type=candidate.heading_type or "semantic",
                        hierarchy_depth=candidate.hierarchy_depth or 1,
                        canonical_section_label=canonical,
                        metadata={"word_count": len(normalized.split())},
                    )
                )
            elif len(normalized.split()) > 10 and canonical is not None:
                malformed += 1
                candidate.metadata["malformed_heading_candidate"] = True
            analyzed.append(candidate)

        analytics = {
            "heading_count": len(headings),
            "malformed_heading_candidates": malformed,
            "heading_confidence_mean": round(sum(item.heading_confidence for item in headings) / max(len(headings), 1), 4),
            "heading_type_distribution": {key: sum(1 for item in headings if item.heading_type == key) for key in {item.heading_type for item in headings}},
        }
        return analyzed, headings, analytics
