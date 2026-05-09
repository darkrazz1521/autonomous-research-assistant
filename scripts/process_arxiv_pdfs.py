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
from autonomous_research_assistant_data.processing.base import ProcessingContext
from autonomous_research_assistant_data.storage.manifest import ManifestStore
from autonomous_research_assistant_data.storage.metadata_store import MetadataStore
from autonomous_research_assistant_data.storage.state import StateStore


def main() -> None:
    parser = build_common_parser("Process arXiv PDFs into cleaned scientific chunks.")
    parser.add_argument("--force-reprocess", action="store_true", help="Ignore manifest hits and rebuild artifacts.")
    parser.add_argument("--repair-only", action="store_true", help="Reuse extracted and cleaned artifacts, then rerun repair onwards.")
    parser.add_argument("--skip-repair", action="store_true", help="Bypass the repair layer and chunk directly from cleaned paragraphs.")
    parser.add_argument("--strict-validation", action="store_true", help="Mark non-ready validation reports as failed.")
    parser.add_argument("--layout-aware", action="store_true", help="Enable structural layout-aware repair passes.")
    parser.add_argument("--dedup-strict", action="store_true", help="Aggressively suppress duplicate paragraphs during repair.")
    parser.add_argument("--equation-repair-level", choices=["conservative", "balanced", "aggressive"], help="Equation repair aggressiveness.")
    parser.add_argument("--disable-column-reconstruction", action="store_true", help="Disable multi-column reconstruction heuristics.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    from autonomous_research_assistant_data.processing.pipeline import PdfProcessingPipeline

    context = ProcessingContext(
        config=config,
        metadata_store=MetadataStore(),
        state_store=StateStore(config.storage.state_dir / "processing_state.json"),
        manifest_store=ManifestStore(config.pdf_processing.manifests_dir / "pdf_processing_manifest.json"),
    )
    stats = PdfProcessingPipeline(
        context,
        force_reprocess=args.force_reprocess,
        repair_only=args.repair_only,
        skip_repair=args.skip_repair,
        strict_validation=args.strict_validation,
        layout_aware=args.layout_aware,
        dedup_strict=args.dedup_strict,
        equation_repair_level=args.equation_repair_level,
        disable_column_reconstruction=args.disable_column_reconstruction,
    ).process()
    get_logger("scripts.process_arxiv_pdfs").info("PDF processing finished", extra={"context": stats})


if __name__ == "__main__":
    main()
