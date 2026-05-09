"""Dataset loading utilities for downstream phases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_from_disk


def load_parquet_split(path: str | Path) -> pd.DataFrame:
    """Load an exported parquet split into a DataFrame."""
    return pd.read_parquet(path)


def load_disk_dataset(path: str | Path):
    """Load a Hugging Face dataset that was saved to disk."""
    return load_from_disk(str(path))

