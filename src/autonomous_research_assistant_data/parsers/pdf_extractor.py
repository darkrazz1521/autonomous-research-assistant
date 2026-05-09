"""Scientific PDF extraction with backend fallbacks and quality scoring."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ExtractedDocument, ExtractedPage


class ScientificPdfExtractor:
    """Extract text from scientific PDFs with backend fallback support."""

    def __init__(self, preferred_backend: str, fallback_backends: list[str]) -> None:
        self.preferred_backend = preferred_backend
        self.fallback_backends = fallback_backends

    def _extract_with_pymupdf(self, pdf_path: Path) -> tuple[list[ExtractedPage], dict]:
        try:
            import fitz  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyMuPDF backend unavailable: install pymupdf/fitz.") from exc

        doc = fitz.open(pdf_path)
        pages: list[ExtractedPage] = []
        font_sizes: list[tuple[float, str]] = []
        for page_index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            page_lines: list[str] = []
            page_blocks: list[str] = []
            block_metadata: list[dict] = []
            raw_dict = page.get_text("dict")
            for block in raw_dict.get("blocks", []):
                block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                block_lines: list[str] = []
                for line in block.get("lines", []):
                    span_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if span_text.strip():
                        page_lines.append(span_text.strip())
                        page_blocks.append(span_text.strip())
                        block_lines.append(span_text.strip())
                    for span in line.get("spans", []):
                        span_value = span.get("text", "").strip()
                        if span_value:
                            font_sizes.append((float(span.get("size", 0.0)), span_value))
                if block_lines:
                    x0, y0, x1, y1 = block_bbox
                    block_metadata.append(
                        {
                            "bbox": [float(x0), float(y0), float(x1), float(y1)],
                            "text": " ".join(block_lines),
                            "line_count": len(block_lines),
                            "avg_line_length": sum(len(item) for item in block_lines) / max(len(block_lines), 1),
                        }
                    )
            pages.append(
                ExtractedPage(
                    page_number=page_index,
                    text="\n".join(item[4].strip() for item in blocks if len(item) >= 5 and item[4].strip()),
                    lines=page_lines,
                    blocks=page_blocks,
                    block_metadata=block_metadata,
                )
            )
        title = max(font_sizes, key=lambda item: item[0])[1] if font_sizes else None
        return pages, {"title": title}

    def _extract_with_pdfplumber(self, pdf_path: Path) -> tuple[list[ExtractedPage], dict]:
        try:
            import pdfplumber  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("pdfplumber backend unavailable: install pdfplumber.") from exc

        pages: list[ExtractedPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                lines = [line.strip() for line in page_text.splitlines() if line.strip()]
                pages.append(
                    ExtractedPage(
                        page_number=page_index,
                        text=page_text,
                        lines=lines,
                        blocks=lines,
                        block_metadata=[],
                    )
                )
        title = pages[0].lines[0] if pages and pages[0].lines else None
        return pages, {"title": title}

    def _quality_score(self, pages: list[ExtractedPage]) -> float:
        full_text = "\n".join(page.text for page in pages)
        total_chars = len(full_text)
        alnum_chars = sum(char.isalnum() for char in full_text)
        alpha_ratio = alnum_chars / max(total_chars, 1)
        populated_pages = sum(1 for page in pages if len(page.text.strip()) > 200)
        page_ratio = populated_pages / max(len(pages), 1)
        long_line_ratio = sum(1 for page in pages for line in page.lines if len(line) > 40) / max(
            sum(len(page.lines) for page in pages),
            1,
        )
        score = (alpha_ratio * 0.4) + (page_ratio * 0.35) + (long_line_ratio * 0.25)
        return round(score, 4)

    def _extract_title(self, pages: list[ExtractedPage], fallback_title: str | None) -> str | None:
        if fallback_title:
            return fallback_title
        if not pages:
            return None
        for line in pages[0].lines[:12]:
            if len(line) > 20 and not re.fullmatch(r"[\d\W_]+", line):
                return line
        return None

    def _extract_formulas(self, pages: list[ExtractedPage]) -> list[str]:
        formulas: list[str] = []
        pattern = re.compile(r"(=|\\sum|\\int|\\alpha|\\beta|≤|≥|≈|\bO\([^)]+\))")
        for page in pages:
            for line in page.lines:
                if pattern.search(line) and len(line) <= 180:
                    formulas.append(line)
        return formulas[:100]

    def _extract_references(self, pages: list[ExtractedPage]) -> list[str]:
        lines = [line for page in pages for line in page.lines]
        references_start = None
        for idx, line in enumerate(lines):
            normalized = line.strip().lower()
            if normalized in {"references", "bibliography"}:
                references_start = idx + 1
                break
        if references_start is None:
            return []
        return lines[references_start : references_start + 80]

    def extract(self, paper_id: str, arxiv_id: str, pdf_path: Path, abstract: str | None = None) -> ExtractedDocument:
        attempts: list[tuple[str, Callable[[Path], tuple[list[ExtractedPage], dict]]]] = []
        backend_map = {
            "pymupdf": self._extract_with_pymupdf,
            "pdfplumber": self._extract_with_pdfplumber,
        }
        ordered_backends = [self.preferred_backend] + [item for item in self.fallback_backends if item != self.preferred_backend]
        for backend_name in ordered_backends:
            if backend_name in backend_map:
                attempts.append((backend_name, backend_map[backend_name]))

        errors: list[str] = []
        for index, (backend_name, backend_func) in enumerate(attempts):
            try:
                pages, extra = backend_func(pdf_path)
                full_text = "\n\n".join(page.text for page in pages if page.text.strip())
                return ExtractedDocument(
                    paper_id=paper_id,
                    arxiv_id=arxiv_id,
                    source_pdf=pdf_path,
                    extraction_backend=backend_name,
                    extraction_fallback_used=index > 0,
                    extraction_quality_score=self._quality_score(pages),
                    title=self._extract_title(pages, extra.get("title")),
                    abstract=abstract,
                    references=self._extract_references(pages),
                    formulas=self._extract_formulas(pages),
                    pages=pages,
                    extracted_text=full_text,
                    page_count=len(pages),
                    processed_at=utc_now(),
                    extra={"attempted_backends": ordered_backends},
                )
            except Exception as exc:
                errors.append(f"{backend_name}: {exc}")
        if errors:
            raise RuntimeError(f"All PDF extraction backends failed for {pdf_path.name}: {' | '.join(errors)}")
        raise RuntimeError(f"No extraction backends configured for {pdf_path.name}")
