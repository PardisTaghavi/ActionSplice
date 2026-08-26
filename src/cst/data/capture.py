"""Portable serialization schema for matched CST teacher trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class TransportCapture:
    """One matched stale-condition/requested-condition trajectory pair."""

    tensors: dict[str, Any]
    metadata: dict[str, Any]


def validate_transport_capture(capture: TransportCapture) -> None:
    """Validate tensor alignment without imposing backend-specific dynamics."""
    import torch

    tensors = capture.tensors
    required = {
        "old_states",
        "new_states",
        "old_predictions",
        "new_predictions",
        "transition_noises",
        "history_tail",
        "old_viewmats",
        "new_viewmats",
        "intrinsics",
        "timesteps",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(f"Transport capture is missing tensors: {missing}")

    denoising_steps = int(capture.metadata["denoising_steps"])
    old_states = tensors["old_states"]
    new_states = tensors["new_states"]
    if old_states.shape != new_states.shape:
        raise ValueError("Old/new state trajectories have different shapes")
    if int(old_states.shape[0]) != denoising_steps:
        raise ValueError("State trajectories must contain z_0,...,z_(K-1)")
    for name in ("old_predictions", "new_predictions"):
        if int(tensors[name].shape[0]) != denoising_steps:
            raise ValueError(f"{name} must contain one prediction per step")
    if int(tensors["transition_noises"].shape[0]) != denoising_steps - 1:
        raise ValueError("Transition-noise trace has the wrong length")
    if int(tensors["timesteps"].numel()) != denoising_steps:
        raise ValueError("Timestep trace has the wrong length")
    if not torch.equal(old_states[0], new_states[0]):
        raise ValueError("Old/new trajectories must start from identical z_0")
    expected_receipts = list(range(1, denoising_steps))
    if list(capture.metadata["receipt_steps"]) != expected_receipts:
        raise ValueError("Receipt-step metadata is inconsistent")


def save_transport_capture(path: Path, capture: TransportCapture) -> None:
    """Persist a capture with all tensors detached on CPU."""
    import torch

    validate_transport_capture(capture)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "metadata": capture.metadata,
            "tensors": {name: tensor.detach().cpu() for name, tensor in capture.tensors.items()},
        },
        path,
    )


def load_transport_capture(path: Path) -> TransportCapture:
    """Load and validate a serialized capture."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported transport capture schema: {path}")
    capture = TransportCapture(
        tensors=dict(payload["tensors"]),
        metadata=dict(payload["metadata"]),
    )
    validate_transport_capture(capture)
    return capture


__all__ = [
    "SCHEMA_VERSION",
    "TransportCapture",
    "load_transport_capture",
    "save_transport_capture",
    "validate_transport_capture",
]
