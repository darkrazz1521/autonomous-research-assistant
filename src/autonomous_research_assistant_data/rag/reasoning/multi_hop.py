"""Iterative retrieval refinement for scientific questions."""

from __future__ import annotations

from collections import OrderedDict

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import RetrievalResult
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi


class MultiHopRetriever:
    """Perform retrieval refinement through lightweight follow-up queries."""

    def __init__(self, config: AppConfig, api: RetrievalApi) -> None:
        self.config = config
        self.api = api

    def _follow_up_queries(self, query: str, first_hop: list[RetrievalResult]) -> list[str]:
        terms: list[str] = []
        for result in first_hop[: self.config.rag.multi_hop.follow_up_queries_per_hop]:
            terms.extend(result.citation_entities[:1])
            terms.extend(result.metadata.get("chunk_topic_signature", [])[:2])
        unique_terms = list(OrderedDict.fromkeys(term for term in terms if term))
        return [f"{query} {term}" for term in unique_terms[: self.config.rag.multi_hop.follow_up_queries_per_hop]]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        hybrid: bool,
        rerank: bool,
        expand_query: bool,
        context_window: bool,
        window_radius: int | None,
    ) -> dict[str, object]:
        first_trace = self.api.search(
            query,
            top_k=top_k,
            mode="hybrid" if hybrid else "dense",
            rerank=rerank,
            expand_query=expand_query,
            context_window=context_window,
            window_radius=window_radius,
        )
        all_results: dict[str, RetrievalResult] = {item.chunk_id: item for item in first_trace.results}
        hops: list[dict[str, object]] = [{"query": query, "result_chunk_ids": list(all_results)}]
        current_queries = self._follow_up_queries(query, first_trace.results)
        for hop_index in range(2, self.config.rag.multi_hop.max_hops + 1):
            if not current_queries:
                break
            next_queries: list[str] = []
            for follow_up in current_queries[: self.config.rag.multi_hop.follow_up_queries_per_hop]:
                trace = self.api.search(
                    follow_up,
                    top_k=self.config.rag.multi_hop.top_k_per_hop,
                    mode="hybrid" if hybrid else "dense",
                    rerank=rerank,
                    expand_query=expand_query,
                    context_window=context_window,
                    window_radius=window_radius,
                )
                for item in trace.results:
                    all_results.setdefault(item.chunk_id, item)
                hops.append({"query": follow_up, "hop": hop_index, "result_chunk_ids": [item.chunk_id for item in trace.results]})
                next_queries.extend(self._follow_up_queries(follow_up, trace.results))
            current_queries = list(OrderedDict.fromkeys(next_queries))
        merged = sorted(all_results.values(), key=lambda item: item.score, reverse=True)
        return {"results": merged, "trace": {"hops": hops, "multi_hop_enabled": True}}
