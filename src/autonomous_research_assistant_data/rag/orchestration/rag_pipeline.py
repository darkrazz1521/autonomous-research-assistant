"""End-to-end Phase 5 RAG pipeline."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import GenerationMetadata, QueryUnderstandingResult, RAGAnswer, RetrievalResult
from autonomous_research_assistant_data.rag.agentic.retrieval_loop import AgenticRetrievalLoop
from autonomous_research_assistant_data.rag.agentic.workflow import AgenticResearchWorkflow
from autonomous_research_assistant_data.rag.answerability.scorer import AnswerabilityScorer
from autonomous_research_assistant_data.rag.answer_quality.scorer import AnswerQualityScorer
from autonomous_research_assistant_data.rag.conversation.session_memory import SessionMemoryStore
from autonomous_research_assistant_data.rag.context_processing.compressor import ContextCompressor
from autonomous_research_assistant_data.rag.generator.service import GenerationService
from autonomous_research_assistant_data.rag.grounding.grounding_checker import GroundingChecker
from autonomous_research_assistant_data.rag.memory.research_memory import ResearchMemoryEnricher
from autonomous_research_assistant_data.rag.observability.reporter import RAGObservabilityReporter
from autonomous_research_assistant_data.rag.prompts.prompt_builder import RetrievalAwarePromptBuilder
from autonomous_research_assistant_data.rag.query_expansion.advanced_expander import AdvancedQueryExpander
from autonomous_research_assistant_data.rag.query_understanding.understanding import QueryUnderstandingAnalyzer
from autonomous_research_assistant_data.rag.reasoning.multi_hop import MultiHopRetriever
from autonomous_research_assistant_data.rag.reranking.precision_reranker import PrecisionReranker
from autonomous_research_assistant_data.rag.retrieval_routing.router import RetrievalRouter
from autonomous_research_assistant_data.rag.structured_generation.answer_builder import StructuredAnswerBuilder
from autonomous_research_assistant_data.rag.synthesis.answer_synthesizer import AnswerSynthesizer
from autonomous_research_assistant_data.rag.validation.hallucination_detection import HallucinationDetector
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi
from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class RAGPipeline:
    """Compose retrieval, prompting, generation, grounding, and memory."""

    def __init__(self, config: AppConfig, retrieval_api: RetrievalApi | None = None) -> None:
        self.config = config
        self.logger = get_logger("rag.orchestration.pipeline")
        self.retrieval_api = retrieval_api or RetrievalApi(config)
        self.prompt_builder = RetrievalAwarePromptBuilder(config)
        self.generator = GenerationService(config)
        self.synthesizer = AnswerSynthesizer(config)
        self.grounding = GroundingChecker(config)
        self.hallucination = HallucinationDetector()
        self.quality = AnswerQualityScorer(config)
        self.multi_hop = MultiHopRetriever(config, self.retrieval_api)
        self.sessions = SessionMemoryStore(config)
        self.memory = ResearchMemoryEnricher()
        self.query_understanding = QueryUnderstandingAnalyzer(config)
        self.query_expander = AdvancedQueryExpander(config)
        self.router = RetrievalRouter(config)
        self.compressor = ContextCompressor(config)
        self.precision_reranker = PrecisionReranker(config)
        self.answerability = AnswerabilityScorer()
        self.structured_builder = StructuredAnswerBuilder()
        self.observability = RAGObservabilityReporter(config)
        self.agentic_retrieval_loop = AgenticRetrievalLoop(config, self.retrieval_api)
        self.agentic_workflow = AgenticResearchWorkflow(config, self.agentic_retrieval_loop)

    def _conversation_id(self, query: str, conversation_id: str | None) -> str:
        if conversation_id:
            return conversation_id
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    def _default_understanding(self, query: str) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            normalized_query=query,
            expanded_terms=[],
            entities=[],
            query_type="definition",
            target_topics=[],
            expected_answer_structure=["definition", "core idea", "evidence", "citations"],
        )

    def _merge_results(self, result_sets: list[list[RetrievalResult]], *, limit: int) -> list[RetrievalResult]:
        merged: "OrderedDict[str, RetrievalResult]" = OrderedDict()
        for results in result_sets:
            for item in results:
                existing = merged.get(item.chunk_id)
                if existing is None or item.score > existing.score:
                    merged[item.chunk_id] = item
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked[:limit]

    def _topic_match_score(self, text: str, terms: list[str]) -> float:
        lowered = text.lower()
        score = 0.0
        for term in terms:
            normalized = term.lower().strip()
            if not normalized:
                continue
            if normalized in lowered:
                score += 1.0 if len(normalized.split()) > 1 else 0.5
        return score

    def _lexical_rescue_results(self, understanding: QueryUnderstandingResult, *, limit: int) -> list[RetrievalResult]:
        terms = [term for term in [*understanding.entities, *understanding.expanded_terms, *understanding.target_topics] if term]
        if not terms:
            return []
        candidates: list[RetrievalResult] = []
        chunk_root = self.config.pdf_processing.chunks_dir
        for path in chunk_root.rglob("*.json"):
            payload = read_json(Path(path), default={})
            for chunk in payload.get("chunks", []):
                chunk_text = str(chunk.get("chunk_text", ""))
                section_name = str(chunk.get("section_name", ""))
                combined = f"{section_name}\n{chunk_text}"
                score = self._topic_match_score(combined, terms)
                if score <= 0:
                    continue
                metadata = {
                    "section_name": section_name,
                    "canonical_section_label": chunk.get("extra", {}).get("canonical_section_label") or chunk.get("canonical_section_label"),
                    "page_range": chunk.get("page_range"),
                    "source_pdf": chunk.get("source_pdf"),
                    "semantic_hash": chunk.get("semantic_hash"),
                    "citation_density": chunk.get("citation_density", 0.0),
                    "equation_density": chunk.get("equation_density", 0.0),
                    "retrieval_quality_score": chunk.get("retrieval_quality_score", 0.0),
                    "retrieval_noise_score": chunk.get("noise_score", 0.0),
                    "coherence_score": chunk.get("coherence_score", 0.0),
                    "structural_integrity_score": chunk.get("structural_integrity_score", 0.0),
                    "narrative_continuity_score": chunk.get("narrative_continuity_score", 0.0),
                    "semantic_boundary_score": chunk.get("semantic_boundary_score", 0.0),
                    "parent_section_id": chunk.get("parent_section_id"),
                    "retrieval_excluded": chunk.get("retrieval_excluded", False),
                    "table_probability": chunk.get("table_probability", 0.0),
                    "benchmark_probability": chunk.get("benchmark_probability", 0.0),
                    "semantic_density_score": chunk.get("semantic_density_score", 0.0),
                    "citation_spans": chunk.get("citation_spans", []),
                    "citation_entities": chunk.get("citation_entities", []),
                    "chunk_topic_signature": chunk.get("chunk_topic_signature", []),
                    "previous_chunk_id": chunk.get("previous_chunk_id"),
                    "next_chunk_id": chunk.get("next_chunk_id"),
                    "source_chunk_path": str(path),
                    "retrieval_source": "lexical_rescue",
                }
                candidates.append(
                    RetrievalResult(
                        chunk_id=str(chunk.get("chunk_id")),
                        paper_id=str(chunk.get("paper_id") or payload.get("paper_id") or chunk.get("arxiv_id")),
                        arxiv_id=str(chunk.get("arxiv_id") or payload.get("arxiv_id") or chunk.get("paper_id")),
                        score=round(0.9 + (score * 0.15), 6),
                        raw_sparse_score=round(score, 6),
                        chunk_text=chunk_text,
                        section_name=section_name,
                        canonical_section_label=metadata["canonical_section_label"],
                        citations=[span.get("text", "") for span in metadata["citation_spans"]],
                        citation_entities=list(metadata["citation_entities"]),
                        neighboring_chunk_ids=[item for item in [metadata["previous_chunk_id"], metadata["next_chunk_id"]] if item],
                        metadata=metadata,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        deduped = self._merge_results([candidates], limit=limit)
        return deduped

    def run(
        self,
        query: str,
        *,
        hybrid: bool = False,
        rerank: bool = False,
        expand_query: bool = False,
        context_window: bool = False,
        multi_hop: bool = False,
        stream: bool = False,
        conversation_id: str | None = None,
        save_session: bool = False,
        max_context_chunks: int | None = None,
        structured_answer: bool = False,
        query_understanding: bool = False,
        mmr: bool = False,
        compression: bool = False,
        hyde: bool = False,
        answerability_filter: bool = False,
        section_routing: bool = False,
        observability: bool = False,
        agentic: bool = False,
        reflection: bool = False,
        iterative_retrieval: bool = False,
        evidence_graph: bool = False,
        refine_answer: bool = False,
        max_reasoning_steps: int | None = None,
        detect_contradictions: bool = False,
    ) -> RAGAnswer:
        normalized_query = self._normalize_query(query)
        resolved_conversation_id = self._conversation_id(normalized_query, conversation_id)
        session = self.sessions.load(resolved_conversation_id) if self.config.rag.memory.enabled else None
        memory_context = self.memory.enrich_query(normalized_query, session)
        enriched_query = str(memory_context["query"])
        understanding_enabled = query_understanding or self.config.rag.query_understanding.enabled
        understanding = self.query_understanding.analyze(enriched_query) if understanding_enabled else self._default_understanding(enriched_query)
        agentic_enabled = agentic or self.config.rag.agentic.enabled
        reflection_enabled = reflection or (agentic_enabled and self.config.rag.agentic.reflection_enabled)
        refinement_enabled = refine_answer or (agentic_enabled and self.config.rag.agentic.refinement_enabled)
        contradiction_enabled = detect_contradictions or (agentic_enabled and self.config.rag.agentic.contradiction_detection_enabled)
        reasoning_steps = min(max_reasoning_steps or self.config.rag.agentic.max_reasoning_steps, self.config.rag.agentic.max_reasoning_steps)
        agentic_plan = self.agentic_workflow.build_plan(normalized_query, understanding) if agentic_enabled else None
        expansion_report = self.query_expander.expand(understanding, enable_hyde=hyde) if expand_query else {
            "original_query": understanding.normalized_query,
            "expanded_terms": understanding.expanded_terms,
            "rewritten_query": understanding.normalized_query,
            "multi_queries": [understanding.normalized_query],
            "hyde_text": "",
        }
        active_query = str(expansion_report["rewritten_query"])
        retrieval_depth = self.router.dynamic_depth(understanding) if (section_routing or self.config.rag.routing.section_routing_enabled) else self.config.retrieval.search.final_top_k
        retrieval_depth = max(retrieval_depth, max_context_chunks or 0, self.config.retrieval.search.final_top_k)
        retrieval_loop_report = None
        seeded_results: list[RetrievalResult] = []
        if agentic_enabled or iterative_retrieval:
            seeded_results, retrieval_loop_report = self.agentic_retrieval_loop.run(
                active_query,
                understanding,
                top_k=retrieval_depth,
                hybrid=hybrid,
                rerank=rerank,
                context_window=context_window,
            )
        if multi_hop or self.config.rag.multi_hop.enabled:
            multi_hop_payload = self.multi_hop.retrieve(
                active_query,
                top_k=retrieval_depth,
                hybrid=hybrid,
                rerank=rerank,
                expand_query=expand_query,
                context_window=context_window,
                window_radius=self.config.retrieval.context_window.radius,
                understanding=understanding,
            )
            results = multi_hop_payload["results"]
            multi_hop_trace = dict(multi_hop_payload["trace"])
            retrieval_metadata = {
                "mode": "hybrid" if hybrid else "dense",
                "active_topics": memory_context.get("active_topics", []),
                "query_expansion_report": expansion_report,
                "query_understanding": understanding.model_dump(mode="json"),
            }
            if retrieval_loop_report is not None:
                retrieval_metadata["retrieval_loop_report"] = retrieval_loop_report.model_dump(mode="json")
        else:
            traces = []
            result_sets: list[list[RetrievalResult]] = []
            candidate_queries = list(expansion_report["multi_queries"])[: (4 if expand_query else 1)]
            if understanding.query_type == "definition" and understanding.expanded_terms:
                candidate_queries.append(" ".join(understanding.expanded_terms[:3]))
            candidate_queries = list(OrderedDict.fromkeys(candidate_queries))
            for candidate_query in candidate_queries:
                trace = self.retrieval_api.search(
                    str(candidate_query),
                    top_k=retrieval_depth,
                    mode="hybrid" if hybrid else "dense",
                    rerank=rerank,
                    expand_query=False,
                    context_window=context_window,
                )
                traces.append(trace)
                result_sets.append(trace.results)
            if not hybrid and understanding.target_topics:
                rescue_query = f"{understanding.normalized_query} {' '.join(understanding.target_topics[:4])}"
                rescue_trace = self.retrieval_api.search(
                    rescue_query,
                    top_k=retrieval_depth,
                    mode="hybrid",
                    rerank=rerank,
                    expand_query=False,
                    context_window=context_window,
                )
                traces.append(rescue_trace)
                result_sets.append(rescue_trace.results)
            lexical_rescue = self._lexical_rescue_results(understanding, limit=retrieval_depth)
            if lexical_rescue:
                result_sets.append(lexical_rescue)
            if seeded_results:
                result_sets.append(seeded_results)
            results = self._merge_results(result_sets, limit=retrieval_depth)
            multi_hop_trace = {
                "hops": [{"query": str(expansion_report["original_query"]), "result_chunk_ids": [item.chunk_id for item in results]}],
                "multi_hop_enabled": False,
            }
            retrieval_metadata = dict(traces[0].metadata if traces else {})
            retrieval_metadata["active_topics"] = memory_context.get("active_topics", [])
            retrieval_metadata["query_expansion_report"] = expansion_report
            retrieval_metadata["query_understanding"] = understanding.model_dump(mode="json")
            retrieval_metadata["candidate_queries"] = candidate_queries
            if retrieval_loop_report is not None:
                retrieval_metadata["retrieval_loop_report"] = retrieval_loop_report.model_dump(mode="json")
        pre_intelligence_scores = [item.score for item in results[:5]]
        if section_routing or self.config.rag.routing.section_routing_enabled:
            results = self.router.reroute(results, understanding)
        if self.config.rag.reranking.enabled:
            results = self.precision_reranker.rerank(results, understanding)
        if answerability_filter:
            results = self.answerability.rank(results, understanding)
            threshold = 0.15
            filtered = [
                item
                for item in results
                if float(item.final_score_breakdown.get("answerability_score", self.answerability.score(item, understanding))) >= threshold
            ]
            if filtered:
                results = filtered
        compressed_results, compression_metrics = self.compressor.compress(
            normalized_query,
            results,
            understanding,
            use_mmr=mmr or self.config.rag.context_processing.mmr_enabled,
            enable_compression=compression or self.config.rag.context_processing.compression_enabled,
        )
        results = compressed_results
        prompt_payload = self.prompt_builder.build(
            normalized_query,
            results,
            understanding=understanding,
            max_context_chunks=max_context_chunks,
        )
        if structured_answer:
            answer_text = self.structured_builder.build(normalized_query, list(prompt_payload["kept_results"]), understanding)
            generation_metadata = GenerationMetadata(
                provider="structured-grounded",
                model_name="structured-grounded-synthesizer",
                backend="heuristic-template",
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.config.rag.generation.max_tokens,
                streaming=False,
                seed=self.config.rag.generation.seed,
                latency_ms=0.0,
                prompt_chars=len(str(prompt_payload["prompt"])),
                completion_chars=len(answer_text),
                extra={
                    "query_type": understanding.query_type,
                    "expected_answer_structure": understanding.expected_answer_structure,
                },
            )
        else:
            answer_text, generation_metadata = self.generator.generate(
                str(prompt_payload["prompt"]),
                metadata={
                    "query": normalized_query,
                    "compressed_context": prompt_payload["compressed_context"],
                    "prompt_type": prompt_payload["prompt_type"],
                    "query_type": understanding.query_type,
                },
                streaming=stream,
            )
        answer = self.synthesizer.synthesize(
            normalized_query,
            answer_text,
            list(prompt_payload["kept_results"]),
            generation_metadata,
            conversation_id=resolved_conversation_id,
            retrieval_metadata=retrieval_metadata,
            multi_hop_trace=multi_hop_trace,
        )
        grounding_report = self.grounding.check(answer)
        hallucination_metrics = self.hallucination.detect(answer, grounding_report)
        quality_report = self.quality.score(answer, grounding_report)
        answer.grounding_report = grounding_report
        answer.answer_quality_report = quality_report
        answer.confidence_score = round(grounding_report.grounding_score * quality_report.overall_answer_quality_score, 6)
        answer.hallucination_score = hallucination_metrics["hallucination_probability"]
        answer.generated_at = utc_now()
        answer.retrieval_metadata.update(hallucination_metrics)
        answer.retrieval_metadata["compression_metrics"] = compression_metrics
        answer.retrieval_metadata["query_understanding"] = understanding.model_dump(mode="json")
        answer.retrieval_metadata["query_expansion_report"] = expansion_report
        answer.retrieval_metadata["structured_answer_enabled"] = structured_answer
        answer.retrieval_metadata["candidate_chunk_ids"] = [item.chunk_id for item in results]
        reflection_report = None
        evidence_graph_report = None
        contradiction_report = None
        if agentic_plan is not None:
            pre_refine_score = answer.answer_quality_report.overall_answer_quality_score if answer.answer_quality_report else 0.0
            answer, reflection_report, evidence_graph_report, contradiction_report = self.agentic_workflow.postprocess(
                answer,
                plan=agentic_plan,
                reflection_enabled=reflection_enabled,
                refine_answer=refinement_enabled,
                detect_contradictions=contradiction_enabled or evidence_graph,
            )
            post_grounding = self.grounding.check(answer)
            post_hallucination = self.hallucination.detect(answer, post_grounding)
            post_quality = self.quality.score(answer, post_grounding)
            answer.grounding_report = post_grounding
            answer.answer_quality_report = post_quality
            answer.confidence_score = round(post_grounding.grounding_score * post_quality.overall_answer_quality_score, 6)
            answer.hallucination_score = post_hallucination["hallucination_probability"]
            answer.retrieval_metadata.update(post_hallucination)
            answer.retrieval_metadata["agentic_enabled"] = True
            answer.retrieval_metadata["reasoning_steps_used"] = min(reasoning_steps, len(agentic_plan.subtasks))
            answer.retrieval_metadata["refinement_gain"] = round((post_quality.overall_answer_quality_score - pre_refine_score), 6)
        if observability or self.config.rag.observability.enabled:
            post_intelligence_scores = [item.score for item in results[:5]]
            reranker_lift = max(sum(post_intelligence_scores) - sum(pre_intelligence_scores), 0.0) / max(len(post_intelligence_scores), 1)
            prompt_efficiency = round(len(str(prompt_payload["compressed_context"])) / max(generation_metadata.prompt_chars, 1), 6)
            observability_report = self.observability.build(
                query=normalized_query,
                retrieval_latency_ms=float(answer.retrieval_metadata.get("latency_ms", 0.0)),
                reranker_lift=reranker_lift,
                chunk_utilization=float(compression_metrics.get("chunk_utilization", 1.0)),
                context_waste_ratio=float(compression_metrics.get("context_waste_ratio", 0.0)),
                prompt_efficiency=prompt_efficiency,
                grounding=answer.grounding_report or grounding_report,
                reasoning_depth=min(reasoning_steps, len(agentic_plan.subtasks)) if agentic_plan is not None else 0,
                retrieval_retries=int(retrieval_loop_report.retries) if retrieval_loop_report is not None else 0,
                refinement_gain=float(answer.retrieval_metadata.get("refinement_gain", 0.0)),
                evidence_graph_nodes=len(evidence_graph_report.nodes) if evidence_graph_report is not None else 0,
                evidence_graph_edges=len(evidence_graph_report.edges) if evidence_graph_report is not None else 0,
                extra_metadata={
                    "agentic_plan_present": agentic_plan is not None,
                    "reflection_enabled": reflection_enabled,
                    "contradiction_enabled": contradiction_enabled,
                    "planning_trace": agentic_plan.model_dump(mode="json") if agentic_plan is not None else {},
                    "retrieval_retry_trace": retrieval_loop_report.model_dump(mode="json") if retrieval_loop_report is not None else {},
                    "reflection_trace": reflection_report.model_dump(mode="json") if reflection_report is not None else {},
                },
            )
            self.observability.write(observability_report)
            answer.retrieval_metadata["observability"] = observability_report.model_dump(mode="json")
        if save_session or self.config.rag.memory.save_sessions_by_default:
            updated_session = self.sessions.append_answer(resolved_conversation_id, answer)
            if agentic_plan is not None:
                self.memory.remember_retrieval(
                    updated_session,
                    {
                        "query": normalized_query,
                        "plan": agentic_plan.model_dump(mode="json"),
                        "retrieval_loop": retrieval_loop_report.model_dump(mode="json") if retrieval_loop_report is not None else {},
                    },
                    max_items=self.config.rag.memory.max_retrieval_history_items,
                )
                self.memory.remember_refinement(
                    updated_session,
                    {
                        "query": normalized_query,
                        "reflection": reflection_report.model_dump(mode="json") if reflection_report is not None else {},
                        "contradictions": contradiction_report.model_dump(mode="json") if contradiction_report is not None else {},
                    },
                    max_items=self.config.rag.memory.max_history_items,
                )
                self.memory.remember_evidence(updated_session, [chunk.chunk_id for chunk in answer.evidence_chunks])
                self.sessions.save(updated_session)
        output_path = self.config.rag.generated_answers_dir / f"{resolved_conversation_id}-{utc_now().strftime('%Y%m%d%H%M%S')}.json"
        write_json(output_path, answer.model_dump(mode="json"))
        cache_key = hashlib.sha256(
            f"{normalized_query}:{hybrid}:{rerank}:{expand_query}:{context_window}:{multi_hop}:{structured_answer}:{query_understanding}:{compression}:{hyde}:{answerability_filter}:{section_routing}:{agentic}:{reflection}:{iterative_retrieval}:{refine_answer}:{detect_contradictions}".encode(
                "utf-8"
            )
        ).hexdigest()
        write_json(self.config.rag.rag_cache_dir / f"{cache_key}.json", answer.model_dump(mode="json"))
        write_json(self.config.rag.rag_outputs_dir / "latest_answer.json", answer.model_dump(mode="json"))
        self.logger.info(
            "Generated RAG answer",
            extra={
                "context": {
                    "conversation_id": resolved_conversation_id,
                    "query": normalized_query,
                    "confidence_score": answer.confidence_score,
                    "hallucination_score": answer.hallucination_score,
                    "citation_count": len(answer.citations),
                }
            },
        )
        return answer
