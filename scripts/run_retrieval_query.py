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
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi


def main() -> None:
    parser = build_common_parser("Run a retrieval query against the scientific corpus.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval instead of dense-only retrieval.")
    parser.add_argument("--rerank", action="store_true", help="Apply reranking to top results.")
    parser.add_argument("--embedding-model", help="Override the configured embedding model.")
    parser.add_argument("--vector-db", help="Override the configured vector store backend.")
    parser.add_argument("--section-filter", help="Filter results by canonical section label.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    api = RetrievalApi(config, model_name=args.embedding_model, backend=args.vector_db)
    trace = api.search(
        args.query,
        top_k=args.top_k,
        mode="hybrid" if args.hybrid else "dense",
        rerank=args.rerank,
        section_filter=args.section_filter,
    )
    print(json.dumps(trace.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
