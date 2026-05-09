from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous_research_assistant_data.bootstrap import bootstrap_directories
from autonomous_research_assistant_data.cli import build_common_parser
from autonomous_research_assistant_data.config import load_config
from autonomous_research_assistant_data.core.logging import configure_logging, get_logger
from autonomous_research_assistant_data.validation.arxiv import ArxivValidator
from autonomous_research_assistant_data.validation.datasets import DatasetValidator


def main() -> None:
    parser = build_common_parser("Validate Phase 2 data artifacts.")
    args = parser.parse_args()

    config = load_config(args.config)
    bootstrap_directories(config)
    configure_logging(config)

    results = {
        "arxiv": ArxivValidator(config).validate(),
        "scifact": DatasetValidator(config, "scifact").validate(),
        "fever": DatasetValidator(config, "fever").validate(),
        "msmarco": DatasetValidator(config, "msmarco").validate(),
    }
    get_logger("scripts.validate_phase2").info("Validation summary", extra={"context": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
