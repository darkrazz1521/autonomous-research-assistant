"""Google Colab runtime helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.core.environment import gpu_runtime_info, is_colab_runtime


def validate_colab_runtime(raise_if_invalid: bool = False) -> bool:
    """Validate whether the current runtime is Google Colab."""
    valid = is_colab_runtime()
    if raise_if_invalid and not valid:
        raise RuntimeError("Google Colab runtime not detected.")
    return valid


def maybe_mount_google_drive(enabled: bool, mount_point: str | Path, force_remount: bool = False) -> bool:
    """Mount Google Drive when enabled inside Colab."""
    if not enabled:
        return False
    validate_colab_runtime(raise_if_invalid=True)
    from google.colab import drive  # type: ignore

    drive.mount(str(mount_point), force_remount=force_remount)
    return True


def google_drive_status(mount_point: str | Path = "/content/drive") -> dict[str, Any]:
    """Return a summary of Google Drive and GPU runtime state."""
    mount_path = Path(mount_point)
    return {
        "is_colab": is_colab_runtime(),
        "mount_point": str(mount_path),
        "mount_exists": mount_path.exists(),
        "gpu": gpu_runtime_info(),
    }
