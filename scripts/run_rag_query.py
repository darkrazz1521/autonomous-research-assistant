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
    )
    print(json.dumps(answer.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
