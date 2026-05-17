"""Citation management for report writing workflows."""

from __future__ import annotations

from collections import OrderedDict
import re

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

    def citation_confidence(self, result: RetrievalResult) -> float:
        quality = float(result.metadata.get("retrieval_quality_score", 0.0))
        density = min(float(result.metadata.get("citation_density", 0.0)) * 5, 1.0)
        integrity = float(result.metadata.get("structural_integrity_score", 0.0))
        return round(min((quality * 0.45) + (density * 0.15) + (integrity * 0.40), 1.0), 6)

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
                "grounding_confidence": self.citation_confidence(result),
                "evidence_traceability": {
                    "source_pdf": result.metadata.get("source_pdf"),
                    "page_range": result.metadata.get("page_range"),
                },
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
            metadata={**result.metadata, "citation_style": self.style, "grounding_confidence": self.citation_confidence(result)},
        )

    def attach_sentence_citation(self, sentence: str, results: list[RetrievalResult]) -> tuple[str, list[RAGCitationRecord]]:
        tokens = set(re.findall(r"[a-z][a-z0-9\-]{2,}", sentence.lower()))
        ranked: list[tuple[float, RetrievalResult]] = []
        for result in results:
            result_tokens = set(re.findall(r"[a-z][a-z0-9\-]{2,}", (result.chunk_text or "").lower()))
            overlap = len(tokens.intersection(result_tokens)) / max(len(tokens), 1)
            ranked.append((overlap + (result.score * 0.1), result))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen = [item[1] for item in ranked[:2] if item[0] > 0]
        merged = self.merge_duplicates([self.build_citation_record(result) for result in chosen])
        if merged:
            labels = " ".join(record.citation_label for record in merged)
            if labels not in sentence:
                sentence = f"{sentence} {labels}"
        return sentence, merged

    def merge_duplicates(self, citations: list[RAGCitationRecord]) -> list[RAGCitationRecord]:
        deduped: OrderedDict[tuple[str, str, str], RAGCitationRecord] = OrderedDict()
        for citation in citations:
            key = (citation.paper_id, citation.section_name, citation.chunk_id)
            deduped.setdefault(key, citation)
        return list(deduped.values())

    def bibliography(self, citations: list[RAGCitationRecord]) -> list[str]:
        merged = self.merge_duplicates(citations)
        return list(dict.fromkeys(citation.bibliography_entry for citation in merged))
