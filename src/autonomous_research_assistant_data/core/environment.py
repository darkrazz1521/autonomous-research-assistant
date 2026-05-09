"""Runtime environment detection helpers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from typing import Literal

EnvironmentName = Literal["local", "colab"]


def is_colab_runtime() -> bool:
    """Return whether the current process is running in Google Colab."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return True
    return importlib.util.find_spec("google.colab") is not None


def detect_runtime_environment() -> EnvironmentName:
    """Detect the active runtime environment."""
    return "colab" if is_colab_runtime() else "local"


def gpu_runtime_info() -> dict[str, str | bool | None]:
    """Inspect basic GPU availability in local and Colab environments."""
    info: dict[str, str | bool | None] = {
        "available": False,
        "backend": None,
        "device_name": None,
    }
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            info["available"] = True
            info["backend"] = "torch"
            info["device_name"] = torch.cuda.get_device_name(0)
            return info
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["available"] = True
            info["backend"] = "nvidia-smi"
            info["device_name"] = result.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return info
