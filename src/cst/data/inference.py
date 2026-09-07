"""Backend-independent single-rollout configuration for CST inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..backends.conditioning import (
    build_event_stale_command_schedules,
    expand_chunk_action_blocks,
)


def load_inference_task(path: Path, *, method: str) -> dict[str, Any]:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    if method not in {"cst_r", "cst_t"}:
        raise ValueError("method must be cst_r or cst_t")
    chunk_size = int(config.get("chunk_size", 4))
    denoising_steps = int(config.get("denoising_steps", 4))
    if chunk_size != 4 or denoising_steps != 4:
        raise ValueError("Release checkpoints require chunk_size=4 and denoising_steps=4")
    blocks = [str(value) for value in config["command_blocks"]]
    if len(blocks) < 2:
        raise ValueError("command_blocks must contain at least two chunks")
    changed_blocks = [
        index for index in range(1, len(blocks)) if blocks[index] != blocks[index - 1]
    ]
    if not changed_blocks:
        raise ValueError("command_blocks must contain at least one action change")
    requested = expand_chunk_action_blocks(blocks, chunk_size)
    stale_all = build_event_stale_command_schedules(blocks, chunk_size)
    events = [index * chunk_size for index in changed_blocks]
    receipts = [int(value) for value in config.get("receipt_steps", [2])]
    if len(receipts) not in {1, len(events)}:
        raise ValueError("receipt_steps must contain one value or one value per event")
    if any(value not in {1, 2, 3} for value in receipts):
        raise ValueError("receipt_steps values must be 1, 2, or 3")
    if len(receipts) == 1:
        receipts *= len(events)

    if method == "cst_t":
        boundaries = [int(value) for value in config.get("boundaries", [2])]
        if len(boundaries) not in {1, len(events)}:
            raise ValueError("boundaries must contain one value or one value per event")
        if any(value not in {1, 2, 3} for value in boundaries):
            raise ValueError("CST-T boundaries must be 1, 2, or 3")
        if len(boundaries) == 1:
            boundaries *= len(events)
    else:
        if "boundaries" in config:
            raise ValueError("CST-R configuration must not define boundaries")
        boundaries = [0] * len(events)

    prompt = str(config.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must be a nonempty string")
    return {
        **config,
        "prompt": prompt,
        "seed": int(config.get("seed", 0)),
        "chunk_size": chunk_size,
        "denoising_steps": denoising_steps,
        "command_blocks": blocks,
        "num_latent_frames": chunk_size * len(blocks),
        "requested_commands": requested,
        "stale_commands_by_event": {event: stale_all[event] for event in events},
        "event_pose_indices": events,
        "receipt_steps": receipts,
        "intra_chunk_offsets_by_event": dict(zip(events, boundaries, strict=True)),
    }


__all__ = ["load_inference_task"]
