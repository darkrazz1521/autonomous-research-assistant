"""Citation management for report writing workflows."""

from __future__ import annotations

from collections import OrderedDict

from autonomous_research_assistant_data.models.common import RAGCitationRecord, RAGEvidenceChunk, RetrievalResult


class CitationManager:
    """Create inline citations, bibliography entries, and traceable evidence links."""

    def __init__(self, style: str = "ieee") -> None:
        self.style = style
        self._paper_numbers: OrderedDict[str, int] = OrderedDict()

    def _paper_index(self, paper_id: str) -> int:
        if paper_id not in self._paper_numbers:
            self._paper_numbers[paper_id] = len(self._paper_numbers) + 1
        return self._paper_numbers[paper_id]

    def format_inline(self, result: RetrievalResult) -> str:
        if self.style == "apa":
            return f"({result.paper_id}, {result.section_name})"
        if self.style == "scientific":
            return f"[{result.paper_id} §{result.section_name}]"
        return f"[{self._paper_index(result.paper_id)}]"

    def format_bibliography_entry(self, result: RetrievalResult) -> str:
        if self.style == "apa":
            return f"{result.paper_id}. {result.section_name}."
        if self.style == "scientific":
            return f"{result.paper_id} - {result.section_name}"
        return f"[{self._paper_index(result.paper_id)}] {result.paper_id} - {result.section_name}"

    def build_citation_record(self, result: RetrievalResult) -> RAGCitationRecord:
        return RAGCitationRecord(
            citation_label=self.format_inline(result),
            paper_id=result.paper_id,
            section_name=result.section_name,
            chunk_id=result.chunk_id,
            bibliography_entry=self.format_bibliography_entry(result),
            metadata={
                "canonical_section_label": result.canonical_section_label,
                "source_chunk_id": result.chunk_id,
                "citation_style": self.style,
            },
        )

    def build_evidence_chunk(self, result: RetrievalResult) -> RAGEvidenceChunk:
        text = result.merged_context or result.chunk_text
        return RAGEvidenceChunk(
            chunk_id=result.chunk_id,
            paper_id=result.paper_id,
            section_name=result.section_name,
            canonical_section_label=result.canonical_section_label,
            quote=text[:900].strip(),
            score=result.score,
            citations=[self.format_inline(result)],
            metadata={**result.metadata, "citation_style": self.style},
        )

    def merge_duplicates(self, citations: list[RAGCitationRecord]) -> list[RAGCitationRecord]:
        deduped: OrderedDict[tuple[str, str, str], RAGCitationRecord] = OrderedDict()
        for citation in citations:
            key = (citation.paper_id, citation.section_name, citation.chunk_id)
            deduped.setdefault(key, citation)
        return list(deduped.values())

    def bibliography(self, citations: list[RAGCitationRecord]) -> list[str]:
        merged = self.merge_duplicates(citations)
        return list(dict.fromkeys(citation.bibliography_entry for citation in merged))

