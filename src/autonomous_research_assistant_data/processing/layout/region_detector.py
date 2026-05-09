"""Layout-aware figure and table region isolation."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import CleanParagraph, LayoutRegion


class RegionIsolationEngine:
    """Separate figure and table regions from semantic narrative text."""

    FIGURE_PATTERN = re.compile(r"^(figure|fig\.)\s+\d+[a-z]?(?:[\s:.\-]|$)", re.IGNORECASE)
    TABLE_PATTERN = re.compile(r"^table\s+\d+[a-z]?(?:[\s:.\-]|$)", re.IGNORECASE)
    INLINE_FIGURE_PATTERN = re.compile(r"^\([a-z]\)\s+", re.IGNORECASE)
    DENSE_NUMERIC_PATTERN = re.compile(r"(?:\b\d+(?:\.\d+)?%?\b[\s,;:/|()]*){5,}", re.IGNORECASE)

    def isolate(self, paper_id: str, arxiv_id: str, paragraphs: list[CleanParagraph]) -> tuple[list[CleanParagraph], list[LayoutRegion], list[LayoutRegion], dict[str, int]]:
        kept: list[CleanParagraph] = []
        figures: list[LayoutRegion] = []
        tables: list[LayoutRegion] = []

        for paragraph in paragraphs:
            candidate = paragraph.model_copy(deep=True)
            text = candidate.text.strip()
            if self.FIGURE_PATTERN.match(text) or self.INLINE_FIGURE_PATTERN.match(text):
                figures.append(
                    LayoutRegion(
                        region_id=f"{paper_id}-fig{len(figures):04d}",
                        paper_id=paper_id,
                        arxiv_id=arxiv_id,
                        page_number=candidate.page_number,
                        region_type="figure",
                        text=text,
                        confidence=0.9 if self.FIGURE_PATTERN.match(text) else 0.76,
                        metadata={"paragraph_id": candidate.paragraph_id},
                    )
                )
                candidate.metadata["isolated_region_type"] = "figure"
                candidate.noise_classifications = list(dict.fromkeys(candidate.noise_classifications + ["figure_region"]))
                if self.INLINE_FIGURE_PATTERN.match(text):
                    continue
            if self.TABLE_PATTERN.match(text) or "table_bleed" in candidate.noise_classifications or self.DENSE_NUMERIC_PATTERN.search(text):
                tables.append(
                    LayoutRegion(
                        region_id=f"{paper_id}-tbl{len(tables):04d}",
                        paper_id=paper_id,
                        arxiv_id=arxiv_id,
                        page_number=candidate.page_number,
                        region_type="table",
                        text=text,
                        confidence=0.88 if self.TABLE_PATTERN.match(text) else 0.74,
                        metadata={"paragraph_id": candidate.paragraph_id},
                    )
                )
                candidate.metadata["isolated_region_type"] = "table"
                candidate.noise_classifications = list(dict.fromkeys(candidate.noise_classifications + ["table_bleed"]))
                if (
                    self.TABLE_PATTERN.match(text)
                    or self.DENSE_NUMERIC_PATTERN.search(text)
                    or len(text.split()) <= 20
                    or candidate.structural_role == "table_block"
                ):
                    continue
            kept.append(candidate)

        return kept, figures, tables, {
            "isolated_figure_count": len(figures),
            "isolated_table_count": len(tables),
        }
