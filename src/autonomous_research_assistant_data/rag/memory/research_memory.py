"""Conversational research memory enrichment."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import ResearchSessionRecord


class ResearchMemoryEnricher:
    """Derive retrieval and prompt hints from prior conversation state."""

    def enrich_query(self, query: str, session: ResearchSessionRecord | None) -> dict[str, object]:
        if session is None:
            return {"query": query, "active_topics": [], "discussed_papers": [], "retrieval_history": [], "evidence_reuse": {}}
        suffix_terms = session.active_research_topics[:3] + session.discussed_papers[:2]
        enriched_query = query if not suffix_terms else f"{query} {' '.join(suffix_terms)}"
        return {
            "query": enriched_query,
            "active_topics": session.active_research_topics,
            "discussed_papers": session.discussed_papers,
            "unresolved_questions": session.unresolved_questions,
            "retrieval_history": session.retrieval_history[-5:],
            "evidence_reuse": session.evidence_reuse,
        }

    def remember_retrieval(self, session: ResearchSessionRecord, payload: dict[str, object], *, max_items: int) -> ResearchSessionRecord:
        session.retrieval_history.append(payload)
        if len(session.retrieval_history) > max_items:
            session.retrieval_history = session.retrieval_history[-max_items:]
        return session

    def remember_refinement(self, session: ResearchSessionRecord, payload: dict[str, object], *, max_items: int) -> ResearchSessionRecord:
        session.refinement_history.append(payload)
        if len(session.refinement_history) > max_items:
            session.refinement_history = session.refinement_history[-max_items:]
        return session

    def remember_evidence(self, session: ResearchSessionRecord, chunk_ids: list[str]) -> ResearchSessionRecord:
        for chunk_id in chunk_ids:
            session.evidence_reuse[chunk_id] = int(session.evidence_reuse.get(chunk_id, 0)) + 1
        return session
