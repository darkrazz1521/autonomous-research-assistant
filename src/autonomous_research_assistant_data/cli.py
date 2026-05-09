"""Shared CLI construction helpers."""

from __future__ import annotations

import argparse

from autonomous_research_assistant_data.config import AppConfig, load_config


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Build a parser with layered environment-aware config options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="Optional override YAML applied on top of base + environment config.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing base/local/colab YAML files.")
    parser.add_argument(
        "--env",
        default="auto",
        choices=["auto", "local", "colab"],
        help="Execution environment. Use auto to detect local vs Colab.",
    )
    return parser


def load_config_from_args(args: argparse.Namespace) -> AppConfig:
    """Load the application config from CLI arguments."""
    return load_config(config_path=args.config, environment=args.env, config_dir=args.config_dir)
