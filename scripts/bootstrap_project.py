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


def main() -> None:
    parser = build_common_parser("Create the Phase 2 directory scaffold.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)
    logger = get_logger("scripts.bootstrap")
    logger.info("Bootstrapped project directories", extra={"context": {"profile": config.profile}})


if __name__ == "__main__":
    main()
