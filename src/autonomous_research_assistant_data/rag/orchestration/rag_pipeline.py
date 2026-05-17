"""End-to-end Phase 5 RAG pipeline."""

from __future__ import annotations

import hashlib

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import RAGAnswer
from autonomous_research_assistant_data.rag.answer_quality.scorer import AnswerQualityScorer
from autonomous_research_assistant_data.rag.conversation.session_memory import SessionMemoryStore
from autonomous_research_assistant_data.rag.generator.service import GenerationService
from autonomous_research_assistant_data.rag.grounding.grounding_checker import GroundingChecker
from autonomous_research_assistant_data.rag.memory.research_memory import ResearchMemoryEnricher
from autonomous_research_assistant_data.rag.prompts.prompt_builder import RetrievalAwarePromptBuilder
from autonomous_research_assistant_data.rag.reasoning.multi_hop import MultiHopRetriever
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

    def _conversation_id(self, query: str, conversation_id: str | None) -> str:
        if conversation_id:
            return conversation_id
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split())

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
    ) -> RAGAnswer:
        normalized_query = self._normalize_query(query)
        resolved_conversation_id = self._conversation_id(normalized_query, conversation_id)
        session = self.sessions.load(resolved_conversation_id) if self.config.rag.memory.enabled else None
        memory_context = self.memory.enrich_query(normalized_query, session)
        enriched_query = str(memory_context["query"])
        if multi_hop or self.config.rag.multi_hop.enabled:
            multi_hop_payload = self.multi_hop.retrieve(
                enriched_query,
                top_k=self.config.retrieval.search.final_top_k,
                hybrid=hybrid,
                rerank=rerank,
                expand_query=expand_query,
                context_window=context_window,
                window_radius=self.config.retrieval.context_window.radius,
            )
            results = multi_hop_payload["results"]
            multi_hop_trace = dict(multi_hop_payload["trace"])
            retrieval_metadata = {"mode": "hybrid" if hybrid else "dense", "active_topics": memory_context.get("active_topics", [])}
        else:
            trace = self.retrieval_api.search(
                enriched_query,
                top_k=self.config.retrieval.search.final_top_k,
                mode="hybrid" if hybrid else "dense",
                rerank=rerank,
                expand_query=expand_query,
                context_window=context_window,
            )
            results = trace.results
            multi_hop_trace = {"hops": [{"query": enriched_query, "result_chunk_ids": [item.chunk_id for item in trace.results]}], "multi_hop_enabled": False}
            retrieval_metadata = dict(trace.metadata)
            retrieval_metadata["active_topics"] = memory_context.get("active_topics", [])
        prompt_payload = self.prompt_builder.build(normalized_query, results, max_context_chunks=max_context_chunks)
        answer_text, generation_metadata = self.generator.generate(
            str(prompt_payload["prompt"]),
            metadata={
                "query": normalized_query,
                "compressed_context": prompt_payload["compressed_context"],
                "prompt_type": prompt_payload["prompt_type"],
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
        if save_session or self.config.rag.memory.save_sessions_by_default:
            self.sessions.append_answer(resolved_conversation_id, answer)
        output_path = self.config.rag.generated_answers_dir / f"{resolved_conversation_id}-{utc_now().strftime('%Y%m%d%H%M%S')}.json"
        write_json(output_path, answer.model_dump(mode="json"))
        cache_key = hashlib.sha256(f"{normalized_query}:{hybrid}:{rerank}:{expand_query}:{context_window}:{multi_hop}".encode("utf-8")).hexdigest()
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
