"""Structured citation parsing for scientific text."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import CitationSpan
from autonomous_research_assistant_data.processing.utils import estimate_token_count, stable_text_hash


class CitationParser:
    """Extract structured citation spans without stripping them from the text."""

    NUMERIC_PATTERN = re.compile(r"\[(?:\d+(?:\s*[-,]\s*\d+)*)\]")
    AUTHOR_YEAR_PAREN_PATTERN = re.compile(
        r"\((?:[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?(?:,\s*[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?)*)\s*,\s*(19|20)\d{2}[a-z]?\)"
    )
    AUTHOR_YEAR_BRACKET_PATTERN = re.compile(
        r"\[(?:[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?(?:,\s*[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?)*)\s*,\s*(19|20)\d{2}[a-z]?\]"
    )
    INLINE_AUTHOR_YEAR_PATTERN = re.compile(
        r"\b([A-Z][A-Za-z'`\-]+(?:\s+et al\.)?)\s+\((19|20)\d{2}[a-z]?\)"
    )
    YEAR_PATTERN = re.compile(r"(19|20)\d{2}[a-z]?")
    ENTITY_PATTERN = re.compile(r"[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?")

    def parse(self, text: str, prefix: str = "cit") -> dict[str, object]:
        spans: list[CitationSpan] = []
        seen_offsets: set[tuple[int, int]] = set()

        for citation_type, pattern in (
            ("numeric", self.NUMERIC_PATTERN),
            ("author_year", self.AUTHOR_YEAR_PAREN_PATTERN),
            ("author_year", self.AUTHOR_YEAR_BRACKET_PATTERN),
            ("author_year_inline", self.INLINE_AUTHOR_YEAR_PATTERN),
        ):
            for match in pattern.finditer(text):
                key = (match.start(), match.end())
                if key in seen_offsets:
                    continue
                citation_text = match.group(0)
                entities = list(dict.fromkeys(entity.strip() for entity in self.ENTITY_PATTERN.findall(citation_text)))
                year_match = self.YEAR_PATTERN.search(citation_text)
                span = CitationSpan(
                    citation_id=f"{prefix}-{len(spans):04d}",
                    text=citation_text,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    citation_type=citation_type,
                    entities=entities,
                    year=int(year_match.group(0)[:4]) if year_match else None,
                    normalized_key=stable_text_hash(citation_text)[:16],
                )
                spans.append(span)
                seen_offsets.add(key)

        spans.sort(key=lambda item: (item.start_offset, item.end_offset))
        entities = list(dict.fromkeys(entity for span in spans for entity in span.entities))
        token_count = estimate_token_count(text)
        density = round(len(spans) / max(token_count, 1), 4)
        offsets = [(span.start_offset, span.end_offset) for span in spans]
        return {
            "citation_spans": spans,
            "citation_density": density,
            "citation_entities": entities,
            "citation_offsets": offsets,
        }
