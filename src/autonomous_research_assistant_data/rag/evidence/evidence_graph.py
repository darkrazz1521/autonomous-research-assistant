"""Lightweight evidence graph construction."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import EvidenceGraphReport, RAGAnswer


class EvidenceGraphBuilder:
    """Build a lightweight graph from answer evidence."""

    def build(self, answer: RAGAnswer) -> EvidenceGraphReport:
        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        for chunk in answer.evidence_chunks:
            nodes.append(
                {
                    "id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "section_name": chunk.section_name,
                    "score": chunk.score,
                }
            )
        for index, left in enumerate(answer.evidence_chunks):
            for right in answer.evidence_chunks[index + 1 :]:
                relation = "same_paper" if left.paper_id == right.paper_id else "cross_paper"
                strength = 0.8 if relation == "same_paper" else 0.5
                edges.append(
                    {
                        "source": left.chunk_id,
                        "target": right.chunk_id,
                        "relation": relation,
                        "support_strength": strength,
                    }
                )
        support_strength = sum(edge["support_strength"] for edge in edges) / max(len(edges), 1) if edges else 0.0
        return EvidenceGraphReport(nodes=nodes, edges=edges, support_strength=round(support_strength, 6))
