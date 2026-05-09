"""Shared CLI construction helpers."""

from __future__ import annotations

import argparse


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Build a parser with the shared config option."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    return parser

