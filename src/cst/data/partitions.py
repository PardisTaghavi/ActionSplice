"""Metadata helpers for prompt-disjoint CST-R/CST-T train/validation splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_capture_metadata(path: Path) -> dict[str, Any]:
    """Read a JSON sidecar, falling back to the tensor payload."""
    path = Path(path).resolve()
    sidecar = path.parent.parent / "metadata" / f"{path.stem}.json"
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Capture has no metadata mapping: {path}")
    return metadata


def capture_group_id(path: Path, *, group_by: str) -> str:
    """Return a stable group ID without exposing prompt text."""
    metadata = load_capture_metadata(path)
    if group_by == "capture":
        return Path(path).stem
    if group_by == "prompt":
        prompt = str(metadata.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Capture is missing prompt metadata: {path}")
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
        return f"prompt_{digest}"
    if group_by not in metadata:
        raise ValueError(f"Capture metadata has no group field {group_by!r}")
    value = str(metadata[group_by]).strip()
    if not value:
        raise ValueError(f"Capture group field {group_by!r} is empty")
    return f"{group_by}_{value}"


__all__ = ["capture_group_id", "load_capture_metadata"]
