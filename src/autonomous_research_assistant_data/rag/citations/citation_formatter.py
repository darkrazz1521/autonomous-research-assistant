"""Citation formatting and bibliography synthesis."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import RAGCitationRecord, RAGEvidenceChunk, RetrievalResult


class CitationFormatter:
    """Format inline citations and bibliography entries from retrieval results."""

    def format_inline(self, result: RetrievalResult) -> str:
        return f"[{result.paper_id} §{result.section_name}]"

    def format_bibliography_entry(self, result: RetrievalResult) -> str:
        return f"{result.paper_id} - {result.section_name}"

    def build_citation_record(self, result: RetrievalResult) -> RAGCitationRecord:
        return RAGCitationRecord(
            citation_label=self.format_inline(result),
            paper_id=result.paper_id,
            section_name=result.section_name,
            chunk_id=result.chunk_id,
            bibliography_entry=self.format_bibliography_entry(result),
            metadata={"canonical_section_label": result.canonical_section_label},
        )

    def build_evidence_chunk(self, result: RetrievalResult) -> RAGEvidenceChunk:
        quote_source = result.merged_context or result.chunk_text
        quote = quote_source[:900].strip()
        return RAGEvidenceChunk(
            chunk_id=result.chunk_id,
            paper_id=result.paper_id,
            section_name=result.section_name,
            canonical_section_label=result.canonical_section_label,
            quote=quote,
            score=result.score,
            citations=[self.format_inline(result)],
            metadata=result.metadata,
        )
