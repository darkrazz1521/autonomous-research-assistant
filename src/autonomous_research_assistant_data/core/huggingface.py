"""Hugging Face runtime compatibility helpers."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from packaging.version import Version

from autonomous_research_assistant_data.config import AppConfig


def installed_version(package_name: str) -> str | None:
    """Return the installed package version if it exists."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def supports_trust_remote_code() -> bool:
    """Return whether the installed datasets package supports trust_remote_code."""
    version = installed_version("datasets")
    if version is None:
        return False
    return Version(version) >= Version("2.16.0")


def huggingface_runtime_report(config: AppConfig) -> dict[str, Any]:
    """Return a compatibility report for the current Hugging Face runtime."""
    datasets_version = installed_version("datasets")
    hub_version = installed_version("huggingface_hub")
    numpy_version = installed_version("numpy")
    pandas_version = installed_version("pandas")
    report: dict[str, Any] = {
        "datasets_version": datasets_version,
        "datasets_target_version": config.huggingface.datasets_version,
        "huggingface_hub_version": hub_version,
        "numpy_version": numpy_version,
        "pandas_version": pandas_version,
        "supports_trust_remote_code": supports_trust_remote_code(),
    }
    if config.huggingface.enable_version_guard and datasets_version:
        report["datasets_target_match"] = Version(datasets_version).base_version == Version(
            config.huggingface.datasets_version
        ).base_version
    return report
