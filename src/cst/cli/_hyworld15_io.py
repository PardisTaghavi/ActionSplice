"""I/O helpers shared by the HY-WM1.5 command-line entry point."""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path
from typing import Any

from ..backends.conditioning import (
    commands_to_hyworld15_actions,
    commands_to_viewmats,
    make_intrinsics,
)
from ..backends.hyworld15 import HYWorldPlayCondition


def condition_from_commands(commands: list[str]) -> HYWorldPlayCondition:
    import torch

    frame_count = len(commands) + 1
    return HYWorldPlayCondition(
        viewmats=torch.from_numpy(commands_to_viewmats(commands)).unsqueeze(0),
        intrinsics=torch.from_numpy(
            make_intrinsics(frame_count, fx=0.5050505, fy=0.89786756)
        ).unsqueeze(0),
        actions=torch.from_numpy(commands_to_hyworld15_actions(commands)).unsqueeze(0),
    )


def decode_latents(pipeline: Any, latents: Any) -> Any:
    import torch
    from hyvideo.commons import auto_offload_model

    latents = latents.detach().to("cpu")
    for name in (
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "byt5_model",
        "vision_encoder",
    ):
        model = getattr(pipeline, name, None)
        if model is not None and hasattr(model, "to"):
            model.to(torch.device("cpu"))
    for name in ("_kv_cache", "_kv_cache_neg"):
        if hasattr(pipeline, name):
            setattr(pipeline, name, [])
    gc.collect()
    torch.cuda.empty_cache()

    pipeline.vae.requires_grad_(False)
    pipeline.vae.disable_spatial_tiling()
    latents = latents.to(pipeline.execution_device)
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)
    if latents.ndim != 5:
        raise ValueError(f"Expected BCTHW latents, got {tuple(latents.shape)}")
    if getattr(pipeline.vae.config, "shift_factor", None):
        latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    else:
        latents = latents / pipeline.vae.config.scaling_factor
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=pipeline.vae_dtype,
            enabled=pipeline.vae_autocast_enabled,
        ),
        auto_offload_model(
            pipeline.vae,
            pipeline.execution_device,
            enabled=pipeline.enable_offloading,
        ),
    ):
        video = pipeline.vae.decode(latents, return_dict=False, generator=None)[0]
    return (video / 2 + 0.5).clamp(0, 1).cpu().float()


def save_video(video: Any, path: Path, *, fps: int) -> None:
    import imageio.v2 as imageio
    import torch

    if video.ndim != 5 or int(video.shape[0]) != 1:
        raise ValueError("Decoded video must have shape [1,C,F,H,W]")
    frames = (
        (video[0] * 255)
        .clamp(0, 255)
        .to(dtype=torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .numpy()
    )
    imageio.mimwrite(path, frames, fps=fps)


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


__all__ = ["condition_from_commands", "decode_latents", "git_revision", "save_video"]
