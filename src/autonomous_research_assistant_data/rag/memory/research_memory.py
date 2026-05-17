"""Conversational research memory enrichment."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import ResearchSessionRecord


class ResearchMemoryEnricher:
    """Derive retrieval and prompt hints from prior conversation state."""

    def enrich_query(self, query: str, session: ResearchSessionRecord | None) -> dict[str, object]:
        if session is None:
            return {"query": query, "active_topics": [], "discussed_papers": []}
        suffix_terms = session.active_research_topics[:3] + session.discussed_papers[:2]
        enriched_query = query if not suffix_terms else f"{query} {' '.join(suffix_terms)}"
        return {
            "query": enriched_query,
            "active_topics": session.active_research_topics,
            "discussed_papers": session.discussed_papers,
            "unresolved_questions": session.unresolved_questions,
        }
