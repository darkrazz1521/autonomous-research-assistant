from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous_research_assistant_data.bootstrap import bootstrap_directories, prepare_runtime
from autonomous_research_assistant_data.cli import build_common_parser, load_config_from_args
from autonomous_research_assistant_data.core.logging import configure_logging, get_logger
from autonomous_research_assistant_data.ingestion.arxiv.pipeline import ArxivIngestor
from autonomous_research_assistant_data.ingestion.base import IngestionContext
from autonomous_research_assistant_data.storage.manifest import ManifestStore
from autonomous_research_assistant_data.storage.metadata_store import MetadataStore
from autonomous_research_assistant_data.storage.state import StateStore


async def async_main(args) -> None:
    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    context = IngestionContext(
        config=config,
        metadata_store=MetadataStore(),
        state_store=StateStore(config.storage.state_dir / "ingestion_state.json"),
        manifest_store=ManifestStore(config.storage.metadata_dir / "manifests" / "arxiv_manifest.json"),
    )
    ingestor = ArxivIngestor(context)
    stats = await ingestor.ingest()
    get_logger("scripts.ingest_arxiv").info("arXiv ingestion finished", extra={"context": stats})


def main() -> None:
    parser = build_common_parser("Run arXiv ingestion.")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
