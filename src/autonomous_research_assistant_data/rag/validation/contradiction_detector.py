"""Contradiction detection across evidence chunks."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import ContradictionReport, EvidenceGraphReport, RAGAnswer


class ContradictionDetector:
    """Heuristically detect conflicting evidence and uncertainty."""

    def detect(self, answer: RAGAnswer, graph: EvidenceGraphReport) -> ContradictionReport:
        disagreement_pairs: list[dict[str, object]] = []
        uncertainty_notes: list[str] = []
        negative_markers = {"however", "whereas", "in contrast", "but", "although"}
        chunks = answer.evidence_chunks
        for index, left in enumerate(chunks):
            left_text = left.quote.lower()
            for right in chunks[index + 1 :]:
                right_text = right.quote.lower()
                left_marked = any(marker in left_text for marker in negative_markers)
                right_marked = any(marker in right_text for marker in negative_markers)
                shared_tokens = set(re.findall(r"[a-z]{4,}", left_text)).intersection(set(re.findall(r"[a-z]{4,}", right_text)))
                if left.paper_id != right.paper_id and shared_tokens and (left_marked or right_marked):
                    disagreement_pairs.append(
                        {
                            "left_chunk_id": left.chunk_id,
                            "right_chunk_id": right.chunk_id,
                            "shared_terms": sorted(shared_tokens)[:6],
                        }
                    )
        score = min(len(disagreement_pairs) / max(len(chunks), 1), 1.0)
        if disagreement_pairs:
            uncertainty_notes.append("Some evidence appears to differ across papers or sections.")
        return ContradictionReport(
            contradiction_score=round(score, 6),
            disagreement_pairs=disagreement_pairs,
            uncertainty_notes=uncertainty_notes,
            metadata={"evidence_graph_edges": len(graph.edges)},
        )
