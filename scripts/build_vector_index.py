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
from autonomous_research_assistant_data.retrieval.embedding.pipeline import EmbeddingPipeline
from autonomous_research_assistant_data.retrieval.vectorstores.builder import VectorIndexBuilder


def main() -> None:
    parser = build_common_parser("Build persistent vector index for scientific chunks.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the vector index from scratch.")
    parser.add_argument("--embedding-model", help="Override the configured embedding model.")
    parser.add_argument("--vector-db", help="Override the configured vector store backend.")
    parser.add_argument("--batch-size", type=int, help="Override the embedding batch size if embeddings must be generated.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    try:
        report = VectorIndexBuilder(config, model_name=args.embedding_model, backend=args.vector_db).build(force_rebuild=args.force_rebuild)
    except FileNotFoundError:
        EmbeddingPipeline(config, model_name=args.embedding_model, batch_size=args.batch_size).generate(force_rebuild=args.force_rebuild)
        report = VectorIndexBuilder(config, model_name=args.embedding_model, backend=args.vector_db).build(force_rebuild=args.force_rebuild)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
