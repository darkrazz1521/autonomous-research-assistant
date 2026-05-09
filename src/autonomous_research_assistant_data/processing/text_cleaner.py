"""Elite scientific text normalization and artifact isolation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections import deque

from autonomous_research_assistant_data.models.common import (
    CleanParagraph,
    ExcludedArtifact,
    ExtractedDocument,
    FrontMatterRecord,
    ReferenceEntry,
)
from autonomous_research_assistant_data.processing.utils import semantic_density_score


class ScientificTextCleaner:
    """Normalize extracted PDF text while preserving scientific and citation structure."""

    FIGURE_TABLE_PATTERN = re.compile(r"^(figure|fig\.|table)\s+\d+[\.:|\s-]", re.IGNORECASE)
    APPENDIX_PATTERN = re.compile(r"^(appendix|supplementary material|supplemental material)\b", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|github\.com/\S+)", re.IGNORECASE)
    WATERMARK_PATTERN = re.compile(r"(preprint|under review|copyright|arxiv:|accepted at)", re.IGNORECASE)
    PAGE_NUMBER_PATTERN = re.compile(r"^(page\s+)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)
    EQUATION_PATTERN = re.compile(r"(=|\\sum|\\int|\\frac|\\alpha|\\beta|≤|≥|≈|\bO\([^)]+\)|\(\d+\))")
    INLINE_CITATION_PATTERN = re.compile(r"(\[[0-9,\-\s]+\]|\([A-Z][A-Za-z]+(?: et al\.)?,?\s+\d{4}\))")
    REFERENCE_ENTRY_PATTERN = re.compile(r"^(\[\d+\]|\d+\.\s|[A-Z][a-z]+,\s[A-Z]\.)")
    HEADING_PATTERN = re.compile(r"^(\d+(\.\d+)*|[IVXLC]+|Appendix\s+[A-Z])[\s\.:]+[A-Z]", re.IGNORECASE)

    def __init__(self, cleaning_rules: dict[str, bool], paragraph_merge_line_threshold: int) -> None:
        self.cleaning_rules = cleaning_rules
        self.paragraph_merge_line_threshold = paragraph_merge_line_threshold

    def _normalize_line(self, line: str) -> str:
        text = unicodedata.normalize("NFKC", line) if self.cleaning_rules.get("normalize_unicode", True) else line
        text = text.replace("\u00ad", "")
        text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[^\S\r\n]+", " ", text)
        return text.strip()

    def _header_footer_candidates(self, document: ExtractedDocument) -> set[str]:
        counter: Counter[str] = Counter()
        for page in document.pages:
            for line in page.lines[:2] + page.lines[-2:]:
                normalized = self._normalize_line(line)
                if 4 <= len(normalized) <= 120:
                    counter[normalized] += 1
        threshold = max(2, int(document.page_count * 0.35))
        return {line for line, count in counter.items() if count >= threshold}

    def _artifact_type(self, line: str, is_front_matter: bool) -> str | None:
        if self.FIGURE_TABLE_PATTERN.match(line):
            return "caption"
        if self.APPENDIX_PATTERN.match(line):
            return "appendix_marker"
        if self.EMAIL_PATTERN.search(line):
            return "email"
        if self.URL_PATTERN.search(line):
            return "link"
        if self.WATERMARK_PATTERN.search(line) and len(line) < 120:
            return "watermark"
        if is_front_matter and ("university" in line.lower() or "institute" in line.lower() or "department" in line.lower()):
            return "affiliation"
        return None

    def _is_noise_line(self, line: str, repeated_lines: set[str]) -> bool:
        if not line:
            return True
        if self.cleaning_rules.get("remove_repeated_headers", True) and line in repeated_lines:
            return True
        if self.cleaning_rules.get("remove_page_numbers", True) and self.PAGE_NUMBER_PATTERN.fullmatch(line):
            return True
        if re.fullmatch(r"[_\W]+", line):
            return True
        return False

    def _looks_like_equation(self, text: str) -> bool:
        return bool(self.EQUATION_PATTERN.search(text)) and (
            len(text.split()) <= 25 or text.count("=") >= 1 or text.count("(") >= 2
        )

    def _starts_new_paragraph(self, previous: str, current: str) -> bool:
        if not previous:
            return False
        if self.HEADING_PATTERN.match(current):
            return True
        if len(previous) > self.paragraph_merge_line_threshold:
            return True
        if current[:1].isupper() and previous.endswith((".", "?", ":", ";")):
            return True
        if current[:1].isdigit() and not current[:4].isdigit():
            return True
        return False

    def _join_lines(self, buffer: list[str]) -> str:
        if not buffer:
            return ""
        merged: list[str] = [buffer[0]]
        for current in buffer[1:]:
            prev = merged[-1]
            if self.cleaning_rules.get("dehyphenate_line_breaks", True) and prev.endswith("-") and current[:1].islower():
                merged[-1] = prev[:-1] + current
                continue
            if self.cleaning_rules.get("smart_newline_reconstruction", True):
                if not prev.endswith((".", "?", "!", ":", ";")) and current[:1].islower():
                    merged[-1] = prev + " " + current
                    continue
                if prev.endswith(",") or prev.endswith("("):
                    merged[-1] = prev + " " + current
                    continue
            merged.append(current)
        text = " ".join(merged)
        if self.cleaning_rules.get("collapse_whitespace", True):
            text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_front_matter(self, document: ExtractedDocument, body_candidates: list[tuple[int, str]]) -> FrontMatterRecord:
        title = document.title
        authors: list[str] = []
        affiliations: list[str] = []
        emails: list[str] = []
        links: list[str] = []
        keywords: list[str] = []

        for _, line in body_candidates[:25]:
            if self.EMAIL_PATTERN.search(line):
                emails.extend(self.EMAIL_PATTERN.findall(line))
            if self.URL_PATTERN.search(line):
                links.extend(self.URL_PATTERN.findall(line))
            lowered = line.lower()
            if "keywords" in lowered:
                keywords.extend([item.strip(" .;") for item in re.split(r"[,;]", line.split(":", 1)[-1]) if item.strip()])
            if ("university" in lowered or "institute" in lowered or "department" in lowered) and line not in affiliations:
                affiliations.append(line)
            if not authors and len(line.split(",")) >= 1 and len(line.split()) <= 20 and "@" not in line:
                comma_segments = [item.strip() for item in re.split(r",| and ", line) if item.strip()]
                if len(comma_segments) >= 2 and all(any(char.isupper() for char in item[:2]) for item in comma_segments):
                    authors = comma_segments

        return FrontMatterRecord(
            title=title,
            authors=authors,
            affiliations=affiliations,
            emails=sorted(set(emails)),
            links=sorted(set(links)),
            abstract=document.abstract,
            keywords=sorted(set(keywords)),
            arxiv_metadata={"extraction_backend": document.extraction_backend},
        )

    def _extract_references(self, lines: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[ReferenceEntry]]:
        references_index = None
        for idx, (_, line) in enumerate(lines):
            lowered = line.strip().lower()
            if lowered in {"references", "bibliography"} or lowered.startswith("references "):
                references_index = idx + 1
                break
        if references_index is None:
            return lines, []

        body = lines[: references_index - 1]
        refs = lines[references_index:]
        entries: list[ReferenceEntry] = []
        buffer: list[str] = []
        page_number = None
        entry_index = 0

        def flush() -> None:
            nonlocal buffer, page_number, entry_index
            if not buffer:
                return
            entries.append(
                ReferenceEntry(
                    reference_id=f"ref-{entry_index:04d}",
                    text=" ".join(buffer).strip(),
                    page_number=page_number,
                )
            )
            entry_index += 1
            buffer = []
            page_number = None

        for ref_page, line in refs:
            if self.REFERENCE_ENTRY_PATTERN.match(line) and buffer:
                flush()
            if page_number is None:
                page_number = ref_page
            buffer.append(line)
        flush()
        return body, entries

    def clean(self, document: ExtractedDocument) -> tuple[str, list[CleanParagraph], dict]:
        repeated_lines = self._header_footer_candidates(document)
        linear_lines: list[tuple[int, str]] = []
        excluded_artifacts: list[ExcludedArtifact] = []
        artifact_index = 0
        removed_lines = 0

        for page in document.pages:
            for raw_line in page.lines:
                normalized = self._normalize_line(raw_line)
                if self._is_noise_line(normalized, repeated_lines):
                    removed_lines += 1
                    continue
                linear_lines.append((page.page_number, normalized))

        front_matter = self._extract_front_matter(document, linear_lines)
        body_lines, references = self._extract_references(linear_lines)

        paragraphs: list[CleanParagraph] = []
        buffer: list[str] = []
        paragraph_page = 1
        paragraph_end_page = 1
        paragraph_index = 0
        recent_lines: set[str] = set()
        recent_line_queue: deque[str] = deque(maxlen=500)

        def flush(is_equation: bool = False, section_hint: str | None = None) -> None:
            nonlocal buffer, paragraph_index, paragraph_page, paragraph_end_page
            text = self._join_lines(buffer)
            if not text:
                buffer = []
                return
            paragraph = CleanParagraph(
                paragraph_id=f"{document.paper_id}-p{paragraph_index:04d}",
                text=text,
                page_number=paragraph_page,
                page_end=paragraph_end_page,
                is_equation=is_equation or self._looks_like_equation(text),
                contains_citation=bool(self.INLINE_CITATION_PATTERN.search(text)),
                section_hint=section_hint,
                semantic_density_score=semantic_density_score(text),
            )
            paragraphs.append(paragraph)
            paragraph_index += 1
            buffer = []

        for idx, (page_number, line) in enumerate(body_lines):
            if not line:
                flush()
                continue

            artifact_type = self._artifact_type(line, is_front_matter=idx < 25)
            if artifact_type and self.cleaning_rules.get("filter_captions", True):
                excluded_artifacts.append(
                    ExcludedArtifact(
                        artifact_id=f"{document.paper_id}-a{artifact_index:04d}",
                        artifact_type=artifact_type,
                        text=line,
                        page_number=page_number,
                        confidence=0.85,
                    )
                )
                artifact_index += 1
                flush()
                continue

            if self.cleaning_rules.get("deduplicate_lines", True) and line in recent_lines and len(line) > 20:
                continue
            recent_lines.add(line)
            recent_line_queue.append(line)
            if len(recent_line_queue) == recent_line_queue.maxlen:
                recent_lines = set(recent_line_queue)

            is_equation = self._looks_like_equation(line)
            if is_equation:
                flush()
                buffer = [line]
                paragraph_page = page_number
                paragraph_end_page = page_number
                flush(is_equation=True)
                continue

            if self.HEADING_PATTERN.match(line) or line.lower() in {"abstract", "introduction", "references"}:
                flush()
                buffer = [line]
                paragraph_page = page_number
                paragraph_end_page = page_number
                flush(section_hint="heading")
                continue

            if not buffer:
                paragraph_page = page_number
            paragraph_end_page = page_number

            if buffer and self._starts_new_paragraph(buffer[-1], line):
                flush()
                paragraph_page = page_number
            buffer.append(line)

        flush()

        cleaned_text = "\n\n".join(item.text for item in paragraphs)
        stats = {
            "paragraph_count": len(paragraphs),
            "removed_lines": removed_lines,
            "repeated_line_count": len(repeated_lines),
            "excluded_artifact_count": len(excluded_artifacts),
            "reference_count": len(references),
            "front_matter_title": front_matter.title,
        }
        return cleaned_text, paragraphs, {
            "stats": stats,
            "front_matter": front_matter.model_dump(mode="json"),
            "excluded_artifacts": [item.model_dump(mode="json") for item in excluded_artifacts],
            "references": [item.model_dump(mode="json") for item in references],
        }
