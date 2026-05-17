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
from autonomous_research_assistant_data.rag.evaluation.rag_evaluator import RAGEvaluator
from autonomous_research_assistant_data.rag.orchestration.rag_pipeline import RAGPipeline
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi


def main() -> None:
    parser = build_common_parser("Evaluate the Phase 5 RAG pipeline.")
    parser.add_argument("--probe-count", type=int, help="Limit the number of evaluation probes.")
    parser.add_argument("--embedding-model", help="Override the configured embedding model.")
    parser.add_argument("--vector-db", help="Override the configured vector store backend.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    retrieval_api = RetrievalApi(config, model_name=args.embedding_model, backend=args.vector_db)
    pipeline = RAGPipeline(config, retrieval_api=retrieval_api)
    evaluator = RAGEvaluator(config, pipeline)
    report = evaluator.evaluate(probe_limit=args.probe_count)
    print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
