"""Citation-grounded answer synthesis."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import GenerationMetadata, RAGAnswer, RetrievalResult
from autonomous_research_assistant_data.rag.citations.citation_formatter import CitationFormatter


class AnswerSynthesizer:
    """Attach citation structure and provenance to generated answers."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.citations = CitationFormatter()

    def synthesize(
        self,
        query: str,
        generated_text: str,
        results: list[RetrievalResult],
        generation_metadata: GenerationMetadata,
        *,
        conversation_id: str | None = None,
        retrieval_metadata: dict[str, object] | None = None,
        multi_hop_trace: dict[str, object] | None = None,
    ) -> RAGAnswer:
        selected = results[: self.config.rag.synthesis.max_evidence_chunks]
        citation_records = [self.citations.build_citation_record(item) for item in selected]
        evidence_chunks = [self.citations.build_evidence_chunk(item) for item in selected]
        answer = generated_text.strip()
        if answer and citation_records:
            sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer) if segment.strip()]
            enriched: list[str] = []
            for index, sentence in enumerate(sentences):
                label = citation_records[min(index, len(citation_records) - 1)].citation_label
                enriched.append(f"{sentence} {label}" if label not in sentence else sentence)
            answer = " ".join(enriched)
        bibliography = list(dict.fromkeys(record.bibliography_entry for record in citation_records))
        return RAGAnswer(
            query=query,
            answer=answer,
            citations=citation_records,
            evidence_chunks=evidence_chunks,
            confidence_score=0.0,
            hallucination_score=0.0,
            retrieval_metadata=dict(retrieval_metadata or {}),
            generation_metadata=generation_metadata,
            conversation_id=conversation_id,
            multi_hop_trace=dict(multi_hop_trace or {}),
            bibliography=bibliography,
        )
