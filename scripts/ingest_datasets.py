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
from autonomous_research_assistant_data.ingestion.datasets.fever import FeverIngestor
from autonomous_research_assistant_data.ingestion.datasets.msmarco import MSMARCOIngestor
from autonomous_research_assistant_data.ingestion.datasets.scifact import SciFactIngestor
from autonomous_research_assistant_data.storage.manifest import ManifestStore
from autonomous_research_assistant_data.storage.metadata_store import MetadataStore
from autonomous_research_assistant_data.storage.state import StateStore


def build_dataset_ingestors(context: IngestionContext) -> dict[str, object]:
    return {
        "scifact": SciFactIngestor(context),
        "fever": FeverIngestor(context),
        "msmarco": MSMARCOIngestor(context),
    }


def main() -> None:
    parser = build_common_parser("Download external benchmark datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "fever", "msmarco"],
        choices=["scifact", "fever", "msmarco"],
        help="Datasets to ingest.",
    )
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    context = IngestionContext(
        config=config,
        metadata_store=MetadataStore(),
        state_store=StateStore(config.storage.state_dir / "ingestion_state.json"),
        manifest_store=ManifestStore(config.storage.metadata_dir / "manifests" / "dataset_manifest.json"),
    )
    ingestors = build_dataset_ingestors(context)

    logger = get_logger("scripts.ingest_datasets")
    for dataset_name in args.datasets:
        if not config.datasets[dataset_name].enabled:
            logger.info("Dataset disabled in config", extra={"context": {"dataset_name": dataset_name}})
            continue
        stats = ingestors[dataset_name].ingest()
        logger.info("Dataset ingestion finished", extra={"context": {"dataset_name": dataset_name, **stats}})


if __name__ == "__main__":
    main()
