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
from autonomous_research_assistant_data.writer.orchestration.writing_workflow import WritingWorkflow


def main() -> None:
    parser = build_common_parser("Run the Phase 6 single-agent research writer.")
    parser.add_argument("--topic", required=True, help="Research topic to write about.")
    parser.add_argument("--report-type", required=True, help="Report type such as literature-review or comparison.")
    parser.add_argument("--style", help="Writer style profile.")
    parser.add_argument("--depth", default="standard", choices=["brief", "standard", "deep"], help="Requested writing depth.")
    parser.add_argument("--max-sections", type=int, help="Maximum number of generated sections.")
    parser.add_argument("--revision-passes", type=int, help="Number of revision passes.")
    parser.add_argument("--agentic", action="store_true", help="Retained for compatibility with agentic report generation workflows.")
    parser.add_argument("--grounded", action="store_true", help="Retained for compatibility with grounded writing workflows.")
    parser.add_argument("--citation-style", help="Citation style: ieee, apa, or scientific.")
    parser.add_argument("--export-format", help="Export format: markdown, json, or structured.")
    parser.add_argument("--embedding-model", help="Override the configured embedding model.")
    parser.add_argument("--vector-db", help="Override the configured vector store backend.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    retrieval_api = RetrievalApi(config, model_name=args.embedding_model, backend=args.vector_db)
    workflow = WritingWorkflow(config, retrieval_api=retrieval_api)
    report = workflow.run(
        topic=args.topic,
        report_type=args.report_type,
        style=args.style,
        depth=args.depth,
        max_sections=args.max_sections,
        revision_passes=args.revision_passes,
        citation_style=args.citation_style,
        export_format=args.export_format,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
