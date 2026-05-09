from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous_research_assistant_data.bootstrap import bootstrap_directories, prepare_runtime
from autonomous_research_assistant_data.cli import build_common_parser, load_config_from_args
from autonomous_research_assistant_data.core.logging import configure_logging, get_logger
from autonomous_research_assistant_data.ingestion.base import IngestionContext
from autonomous_research_assistant_data.storage.manifest import ManifestStore
from autonomous_research_assistant_data.storage.metadata_store import MetadataStore
from autonomous_research_assistant_data.storage.state import StateStore


def main() -> None:
    parser = build_common_parser("Run conservative Colab-safe arXiv ingestion.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    from autonomous_research_assistant_data.ingestion.arxiv.simple import SimpleArxivIngestor

    context = IngestionContext(
        config=config,
        metadata_store=MetadataStore(),
        state_store=StateStore(config.storage.state_dir / "ingestion_state.json"),
        manifest_store=ManifestStore(config.storage.metadata_dir / "manifests" / "arxiv_manifest.json"),
    )
    stats = SimpleArxivIngestor(context).ingest()
    get_logger("scripts.ingest_arxiv_simple").info("Simple arXiv ingestion finished", extra={"context": stats})


if __name__ == "__main__":
    main()
