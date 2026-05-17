"""Section-specific retrieval orchestration for report writing."""

from __future__ import annotations

from collections import OrderedDict

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import ContradictionReport, QueryUnderstandingResult, RetrievalResult, SectionEvidenceRecord, WritingSectionPlan, WriterSessionRecord
from autonomous_research_assistant_data.rag.agentic.retrieval_loop import AgenticRetrievalLoop
from autonomous_research_assistant_data.rag.query_expansion.advanced_expander import AdvancedQueryExpander
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi


class SectionOrchestrator:
    """Gather evidence for one report section using the existing retrieval stack."""

    def __init__(self, config: AppConfig, retrieval_api: RetrievalApi | None = None) -> None:
        self.config = config
        self.retrieval_api = retrieval_api or RetrievalApi(config)
        self.expander = AdvancedQueryExpander(config)
        self.retrieval_loop = AgenticRetrievalLoop(config, self.retrieval_api)

    def _merge(self, result_sets: list[list[RetrievalResult]], limit: int) -> list[RetrievalResult]:
        merged: OrderedDict[str, RetrievalResult] = OrderedDict()
        for results in result_sets:
            for item in results:
                current = merged.get(item.chunk_id)
                if current is None or item.score > current.score:
                    merged[item.chunk_id] = item
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked[:limit]

    def _coverage(self, results: list[RetrievalResult], required_terms: list[str]) -> float:
        if not results or not required_terms:
            return 0.0
        covered = 0
        terms = [term.lower() for term in required_terms if term]
        for term in terms:
            if any(term in f"{item.section_name} {item.chunk_text}".lower() for item in results[:6]):
                covered += 1
        return round(covered / max(len(terms), 1), 6)

    def _contradictions(self, results: list[RetrievalResult]) -> ContradictionReport:
        pairs: list[dict[str, object]] = []
        markers = ("however", "whereas", "in contrast", "but", "although")
        for index, left in enumerate(results[:6]):
            left_text = left.chunk_text.lower()
            for right in results[index + 1 : 6]:
                right_text = right.chunk_text.lower()
                if left.paper_id == right.paper_id:
                    continue
                if any(marker in left_text or marker in right_text for marker in markers):
                    shared = sorted(set(left.citation_entities).intersection(set(right.citation_entities)))[:4]
                    pairs.append({"left_chunk_id": left.chunk_id, "right_chunk_id": right.chunk_id, "shared_terms": shared})
        score = round(min(len(pairs) / max(len(results[:6]), 1), 1.0), 6)
        notes = ["Cross-paper evidence contains contrast markers that may require careful synthesis."] if pairs else []
        return ContradictionReport(contradiction_score=score, disagreement_pairs=pairs, uncertainty_notes=notes)

    def orchestrate(
        self,
        topic: str,
        plan: WritingSectionPlan,
        understanding: QueryUnderstandingResult,
        session: WriterSessionRecord | None = None,
    ) -> SectionEvidenceRecord:
        expanded = self.expander.expand(understanding, enable_hyde=False)
        retrieval_results, loop_report = self.retrieval_loop.run(
            plan.section_query,
            understanding,
            top_k=max(self.config.writer.max_section_context_chunks, self.config.retrieval.search.final_top_k),
            hybrid=True,
            rerank=True,
            context_window=True,
        )
        result_sets = [retrieval_results]
        for query in list(expanded["multi_queries"])[:3]:
            section_query = f"{plan.title} {query}"
            trace = self.retrieval_api.search(
                section_query,
                top_k=self.config.writer.max_section_context_chunks,
                mode="hybrid",
                rerank=True,
                expand_query=False,
                context_window=True,
            )
            result_sets.append(trace.results)
        if session and session.sections:
            prior_summary = session.sections[-1].summary
            if prior_summary:
                trace = self.retrieval_api.search(
                    f"{plan.section_query} {prior_summary}",
                    top_k=max(4, self.config.writer.max_section_context_chunks // 2),
                    mode="hybrid",
                    rerank=True,
                    expand_query=False,
                    context_window=True,
                )
                result_sets.append(trace.results)
        merged = self._merge(result_sets, limit=self.config.writer.max_section_context_chunks)
        coverage = self._coverage(merged, plan.required_terms)
        contradictions = self._contradictions(merged)
        return SectionEvidenceRecord(
            section_id=plan.section_id,
            query=plan.section_query,
            retrieval_results=merged,
            coverage_score=coverage,
            contradiction_report=contradictions,
            retrieval_loop_report=loop_report,
            metadata={
                "section_title": plan.title,
                "retrieval_strategy": plan.retrieval_strategy,
                "dependency_count": len(plan.dependencies),
                "required_terms": plan.required_terms,
            },
        )

