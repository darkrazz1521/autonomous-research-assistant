"""Structured logging setup."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.config import AppConfig


class JsonFormatter(logging.Formatter):
    """Serialize log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "context") and isinstance(record.context, dict):
            payload["context"] = record.context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(config: AppConfig) -> None:
    """Configure root logging once for the application."""
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))

    formatter = JsonFormatter()

    _ensure_parent(config.logging.ingestion_log_file)
    ingestion_handler = logging.FileHandler(config.logging.ingestion_log_file, encoding="utf-8")
    ingestion_handler.setFormatter(formatter)

    _ensure_parent(config.logging.failure_log_file)
    failure_handler = logging.FileHandler(config.logging.failure_log_file, encoding="utf-8")
    failure_handler.setLevel(logging.WARNING)
    failure_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(ingestion_handler)
    root_logger.addHandler(failure_handler)

    if config.logging.console_enabled:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    """Return a named application logger."""
    return logging.getLogger(name)
