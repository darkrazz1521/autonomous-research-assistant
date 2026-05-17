from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous_research_assistant_data.bootstrap import bootstrap_directories, prepare_runtime
from autonomous_research_assistant_data.cli import build_common_parser, load_config_from_args
from autonomous_research_assistant_data.core.logging import configure_logging
from autonomous_research_assistant_data.rag.orchestration.rag_pipeline import RAGPipeline
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi


def main() -> None:
    parser = build_common_parser("Run a Phase 5 RAG query against the scientific corpus.")
    parser.add_argument("--query", required=True, help="Research question to answer.")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval.")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking.")
    parser.add_argument("--expand-query", action="store_true", help="Enable query expansion.")
    parser.add_argument("--context-window", action="store_true", help="Enable context-window retrieval enrichment.")
    parser.add_argument("--multi-hop", action="store_true", help="Enable multi-hop retrieval refinement.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming-friendly generation mode.")
    parser.add_argument("--conversation-id", help="Reuse a persistent conversation session id.")
    parser.add_argument("--save-session", action="store_true", help="Persist conversation memory after answer generation.")
    parser.add_argument("--max-context-chunks", type=int, help="Override the maximum number of context chunks passed to prompting.")
    parser.add_argument("--structured-answer", action="store_true", help="Use structured grounded synthesis instead of plain generation.")
    parser.add_argument("--query-understanding", action="store_true", help="Enable semantic query understanding.")
    parser.add_argument("--mmr", action="store_true", help="Enable MMR-style context selection.")
    parser.add_argument("--compression", action="store_true", help="Enable context compression and deduplication.")
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE-style hypothetical query expansion.")
    parser.add_argument("--answerability-filter", action="store_true", help="Prioritize direct-answer evidence before prompting.")
    parser.add_argument("--section-routing", action="store_true", help="Enable section-aware routing by query intent.")
    parser.add_argument("--observability", action="store_true", help="Write RAG observability metrics for this run.")
    parser.add_argument("--agentic", action="store_true", help="Enable the agentic research workflow.")
    parser.add_argument("--reflection", action="store_true", help="Enable answer self-reflection and critique.")
    parser.add_argument("--iterative-retrieval", action="store_true", help="Enable adaptive retrieval retries.")
    parser.add_argument("--evidence-graph", action="store_true", help="Build evidence graph traces for the answer.")
    parser.add_argument("--refine-answer", action="store_true", help="Run answer refinement after synthesis.")
    parser.add_argument("--max-reasoning-steps", type=int, help="Override the maximum bounded reasoning depth.")
    parser.add_argument("--detect-contradictions", action="store_true", help="Enable contradiction detection across evidence.")
    parser.add_argument("--embedding-model", help="Override the configured embedding model.")
    parser.add_argument("--vector-db", help="Override the configured vector store backend.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    retrieval_api = RetrievalApi(config, model_name=args.embedding_model, backend=args.vector_db)
    pipeline = RAGPipeline(config, retrieval_api=retrieval_api)
    answer = pipeline.run(
        args.query,
        hybrid=args.hybrid,
        rerank=args.rerank,
        expand_query=args.expand_query,
        context_window=args.context_window,
        multi_hop=args.multi_hop,
        stream=args.stream,
        conversation_id=args.conversation_id,
        save_session=args.save_session,
        max_context_chunks=args.max_context_chunks,
        structured_answer=args.structured_answer,
        query_understanding=args.query_understanding,
        mmr=args.mmr,
        compression=args.compression,
        hyde=args.hyde,
        answerability_filter=args.answerability_filter,
        section_routing=args.section_routing,
        observability=args.observability,
        agentic=args.agentic,
        reflection=args.reflection,
        iterative_retrieval=args.iterative_retrieval,
        evidence_graph=args.evidence_graph,
        refine_answer=args.refine_answer,
        max_reasoning_steps=args.max_reasoning_steps,
        detect_contradictions=args.detect_contradictions,
    )
    print(json.dumps(answer.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
