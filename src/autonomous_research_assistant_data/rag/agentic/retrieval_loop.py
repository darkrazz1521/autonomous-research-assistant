"""Adaptive retrieval retry loop."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalLoopReport, RetrievalResult
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi


class AgenticRetrievalLoop:
    """Retry retrieval with bounded query reformulation when evidence is weak."""

    def __init__(self, config: AppConfig, retrieval_api: RetrievalApi) -> None:
        self.config = config
        self.retrieval_api = retrieval_api

    def _coverage(self, results: list[RetrievalResult], understanding: QueryUnderstandingResult) -> float:
        if not results:
            return 0.0
        target_terms = understanding.target_topics[:6] + understanding.entities[:3]
        covered = 0
        for result in results[:5]:
            text = f"{result.section_name} {result.chunk_text}".lower()
            if any(term.lower() in text for term in target_terms if term):
                covered += 1
        return covered / max(min(len(results), 5), 1)

    def _retry_queries(self, query: str, understanding: QueryUnderstandingResult) -> list[str]:
        retries = [query]
        if understanding.expanded_terms:
            retries.append(f"{query} {' '.join(understanding.expanded_terms[:4])}")
        if understanding.entities:
            retries.append(" ".join(understanding.entities[:2]))
        if understanding.target_topics:
            retries.append(" ".join(understanding.target_topics[:4]))
        return list(dict.fromkeys(item.strip() for item in retries if item.strip()))

    def run(
        self,
        query: str,
        understanding: QueryUnderstandingResult,
        *,
        top_k: int,
        hybrid: bool,
        rerank: bool,
        context_window: bool,
    ) -> tuple[list[RetrievalResult], RetrievalLoopReport]:
        retry_queries = self._retry_queries(query, understanding)
        best_results: list[RetrievalResult] = []
        best_coverage = 0.0
        retries_used = 0
        for candidate in retry_queries[: self.config.rag.agentic.retrieval_retry_limit + 1]:
            trace = self.retrieval_api.search(
                candidate,
                top_k=top_k,
                mode="hybrid" if hybrid else "dense",
                rerank=rerank,
                expand_query=False,
                context_window=context_window,
            )
            coverage = self._coverage(trace.results, understanding)
            if coverage >= best_coverage:
                best_results = trace.results
                best_coverage = coverage
            if coverage >= 0.6:
                return best_results, RetrievalLoopReport(
                    query=query,
                    retries=retries_used,
                    retrieval_confidence=round(coverage, 6),
                    stopping_reason="coverage_satisfied",
                    retry_queries=retry_queries[: retries_used + 1],
                    coverage_score=round(coverage, 6),
                )
            retries_used += 1
        return best_results, RetrievalLoopReport(
            query=query,
            retries=max(retries_used - 1, 0),
            retrieval_confidence=round(best_coverage, 6),
            stopping_reason="retry_limit_reached",
            retry_queries=retry_queries[:retries_used],
            coverage_score=round(best_coverage, 6),
        )
