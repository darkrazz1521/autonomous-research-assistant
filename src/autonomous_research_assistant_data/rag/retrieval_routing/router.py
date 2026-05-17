"""Route and rescore retrieved results by query intent and section relevance."""

from __future__ import annotations

from collections import defaultdict

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult


SECTION_PRIORS = {
    "definition": {"abstract": 0.20, "introduction": 0.18, "methodology": 0.12, "methods": 0.12},
    "comparison": {"results": 0.20, "discussion": 0.16, "experiments": 0.18, "methodology": 0.10},
    "summarization": {"abstract": 0.18, "introduction": 0.14, "conclusion": 0.14},
    "contradiction_analysis": {"results": 0.18, "discussion": 0.16, "related_work": 0.10},
    "methodology_explanation": {"methodology": 0.22, "methods": 0.22, "introduction": 0.10},
    "benchmark_performance": {"results": 0.22, "discussion": 0.16, "abstract": 0.08},
    "citation_lookup": {"related_work": 0.16, "introduction": 0.12},
    "literature_review": {"related_work": 0.20, "introduction": 0.12, "discussion": 0.08},
    "timeline_history": {"introduction": 0.14, "related_work": 0.18, "conclusion": 0.08},
}


class RetrievalRouter:
    """Intent-aware routing and dynamic retrieval depth selection."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def dynamic_depth(self, understanding: QueryUnderstandingResult) -> int:
        if understanding.query_type == "definition":
            return self.config.rag.routing.definition_depth
        if understanding.query_type == "comparison":
            return self.config.rag.routing.comparison_depth
        if understanding.query_type == "literature_review":
            return self.config.rag.routing.literature_review_depth
        return self.config.retrieval.search.final_top_k

    def reroute(self, results: list[RetrievalResult], understanding: QueryUnderstandingResult) -> list[RetrievalResult]:
        priors = SECTION_PRIORS.get(understanding.query_type, {})
        paper_counts: dict[str, int] = defaultdict(int)
        for result in results[: self.config.rag.routing.paper_cluster_limit]:
            paper_counts[result.paper_id] += 1
        routed: list[RetrievalResult] = []
        for result in results:
            updated = result.model_copy(deep=True)
            label = (updated.canonical_section_label or "").lower()
            section_bonus = priors.get(label, 0.0)
            topic_bonus = 0.0
            paper_bonus = min(paper_counts.get(updated.paper_id, 0), 3) * 0.01
            text = (updated.chunk_text + " " + updated.section_name).lower()
            if any(topic.lower() in text for topic in understanding.target_topics[:6]):
                topic_bonus += 0.08
            if understanding.query_type == "benchmark_performance" and updated.metadata.get("benchmark_probability", 0.0) > 0.2:
                topic_bonus += 0.06
            updated.score = round(updated.score + section_bonus + topic_bonus + paper_bonus, 6)
            updated.final_score_breakdown["routing_section_bonus"] = section_bonus
            updated.final_score_breakdown["routing_topic_bonus"] = topic_bonus
            updated.final_score_breakdown["routing_paper_bonus"] = paper_bonus
            routed.append(updated)
        routed.sort(key=lambda item: item.score, reverse=True)
        return routed
