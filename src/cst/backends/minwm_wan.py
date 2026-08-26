"""Thin, external adapter around an unmodified minWM checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

from .conditioning import commands_to_viewmats, make_intrinsics


class ActionInjectionHook(AbstractContextManager["ActionInjectionHook"]):
    """Swap the active camera trajectory between minWM denoising calls."""

    def __init__(
        self,
        *,
        pipeline: Any,
        desired_viewmats: Any,
        desired_intrinsics: Any,
        event_pose_index: int,
        switch_after_denoising_steps: int | None,
        denoising_steps: int,
    ) -> None:
        self.pipeline = pipeline
        self.generator = pipeline.generator
        self.desired_viewmats = desired_viewmats
        self.desired_intrinsics = desired_intrinsics
        self.event_pose_index = event_pose_index
        self.switch_after_denoising_steps = switch_after_denoising_steps
        self.denoising_steps = denoising_steps
        self.original_forward: Callable[..., Any] | None = None
        self.call_counts: dict[int, int] = {}
        self.events: list[dict[str, Any]] = []
        self.wall_origin: float | None = None

    def __enter__(self) -> "ActionInjectionHook":
        import torch

        self.original_forward = self.generator.forward
        self.wall_origin = time.perf_counter()
        original_forward = self.original_forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            start_token = int(kwargs.get("current_start", 0))
            start_frame = start_token // int(self.pipeline.frame_seq_length)
            call_index = self.call_counts.get(start_frame, 0)
            self.call_counts[start_frame] = call_index + 1
            frame_count = int(kwargs["noisy_image_or_video"].shape[1])

            if start_frame > self.event_pose_index:
                use_desired = True
            elif start_frame < self.event_pose_index:
                use_desired = False
            elif self.switch_after_denoising_steps is None:
                use_desired = False
            else:
                use_desired = call_index >= self.switch_after_denoising_steps

            if use_desired:
                frame_slice = slice(start_frame, start_frame + frame_count)
                kwargs["viewmats"] = self.desired_viewmats[:, frame_slice]
                kwargs["Ks"] = self.desired_intrinsics[:, frame_slice]

            phase = (
                f"denoise_{call_index + 1}"
                if call_index < self.denoising_steps
                else "kv_context_update"
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            relative_start_ms = (
                (started - self.wall_origin) * 1000.0 if self.wall_origin is not None else None
            )
            result = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
            ended = time.perf_counter()
            elapsed_ms = (ended - started) * 1000.0

            timestep = kwargs.get("timestep")
            timestep_value = None
            if timestep is not None:
                timestep_value = int(timestep.flatten()[0].detach().cpu().item())
            self.events.append(
                {
                    "start_frame": start_frame,
                    "frame_count": frame_count,
                    "call_index": call_index,
                    "phase": phase,
                    "used_desired_action": use_desired,
                    "timestep": timestep_value,
                    "relative_start_ms": relative_start_ms,
                    "relative_end_ms": (
                        (ended - self.wall_origin) * 1000.0
                        if self.wall_origin is not None
                        else None
                    ),
                    "elapsed_ms": elapsed_ms,
                }
            )
            return result

        self.generator.forward = wrapped_forward
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.original_forward is not None:
            self.generator.forward = self.original_forward
        return None


def _prepend_minwm_paths(minwm_root: Path) -> None:
    for path in (minwm_root, minwm_root / "Wan21", minwm_root / "shared"):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def minwm_commit(minwm_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=minwm_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_wan_pipeline(
    minwm_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    low_memory: bool | None = None,
) -> tuple[Any, Any, Any]:
    """Load minWM's Wan four-step pipeline once for a job shard.

    Low-memory mode keeps UMT5-XXL on CPU and moves parameters on demand using
    minWM's DynamicSwapInstaller. If unspecified, it is enabled below 45 GiB
    free VRAM. Prompt embeddings are cached for the lifetime of the pipeline.
    """
    import torch
    from omegaconf import OmegaConf

    minwm_root = minwm_root.resolve()
    _prepend_minwm_paths(minwm_root)
    os.chdir(minwm_root)

    from pipeline import CausalInferencePipeline

    config = OmegaConf.load(str(config_path))
    default_config = OmegaConf.load(str(minwm_root / "Wan21/configs/default_config.yaml"))
    config = OmegaConf.merge(default_config, config)
    if len(config.denoising_step_list) != 4:
        raise ValueError("Phase 1 requires a four-step checkpoint/configuration")

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.set_grad_enabled(False)
    free_vram_gb = torch.cuda.mem_get_info(device)[0] / (1024**3)
    use_low_memory = free_vram_gb < 45.0 if low_memory is None else low_memory
    pipeline = CausalInferencePipeline(config, device=device)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator_state = state_dict.get("generator_ema", state_dict.get("generator"))
    if generator_state is None:
        raise KeyError("Checkpoint contains neither generator_ema nor generator")
    try:
        pipeline.generator.load_state_dict(generator_state)
    except RuntimeError:
        fixed = {
            key.replace("model._fsdp_wrapped_module.", "model.", 1)
            if key.startswith("model._fsdp_wrapped_module.")
            else key: value
            for key, value in generator_state.items()
        }
        pipeline.generator.load_state_dict(fixed, strict=False)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if use_low_memory:
        from demo_utils import memory as memory_utils

        memory_utils.gpu = device
        memory_utils.DynamicSwapInstaller.install_model(
            pipeline.text_encoder,
            device=device,
        )
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    prompt_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    prompt_cache_stats = {"hits": 0, "misses": 0}
    original_text_forward = pipeline.text_encoder.forward

    def cached_text_forward(text_prompts: list[str]) -> dict[str, Any]:
        key = tuple(str(prompt) for prompt in text_prompts)
        cached = prompt_cache.get(key)
        if cached is not None:
            prompt_cache_stats["hits"] += 1
            return cached
        prompt_cache_stats["misses"] += 1
        encoded = original_text_forward(text_prompts=text_prompts)
        prompt_cache[key] = encoded
        return encoded

    pipeline.text_encoder.forward = cached_text_forward
    pipeline.reactive_low_memory = use_low_memory
    pipeline.reactive_initial_free_vram_gb = free_vram_gb
    pipeline.reactive_prompt_cache = prompt_cache
    pipeline.reactive_prompt_cache_stats = prompt_cache_stats
    resident_vram_gb = torch.cuda.memory_allocated(device) / (1024**3)
    print(
        "Reactive WM loader: "
        f"device={torch.cuda.get_device_name(device)}, "
        f"initial_free_vram_gb={free_vram_gb:.2f}, "
        f"low_memory={use_low_memory}, "
        f"resident_cuda_memory_gb={resident_vram_gb:.2f}"
    )
    return pipeline, config, device


def run_task(
    *,
    pipeline: Any,
    task: dict[str, Any],
    output_dir: Path,
    minwm_revision: str,
    checkpoint_path: Path,
    model_config_path: Path,
    experiment_config_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    import torch
    from torchvision.io import write_video
    from wan_utils.misc import set_seed

    video_dir = output_dir / "videos"
    metadata_dir = output_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{task['task_id']}.mp4"
    metadata_path = metadata_dir / f"{task['task_id']}.json"
    if video_path.exists() and metadata_path.exists() and not overwrite:
        return {"task_id": task["task_id"], "status": "skipped", "video_path": str(video_path)}

    set_seed(int(task["seed"]))
    device = next(pipeline.generator.parameters()).device
    dtype = torch.bfloat16
    num_frames = int(task["num_latent_frames"])
    stale_np = commands_to_viewmats(task["stale_commands"])
    desired_np = commands_to_viewmats(task["desired_commands"])
    intrinsics_np = make_intrinsics(num_frames)
    stale_viewmats = torch.from_numpy(stale_np).unsqueeze(0).to(device=device, dtype=dtype)
    desired_viewmats = torch.from_numpy(desired_np).unsqueeze(0).to(device=device, dtype=dtype)
    intrinsics = torch.from_numpy(intrinsics_np).unsqueeze(0).to(device=device, dtype=dtype)
    noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=dtype,
    )

    hook = ActionInjectionHook(
        pipeline=pipeline,
        desired_viewmats=desired_viewmats,
        desired_intrinsics=intrinsics,
        event_pose_index=int(task["event_pose_index"]),
        switch_after_denoising_steps=task["switch_after_denoising_steps"],
        denoising_steps=int(task["denoising_steps"]),
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    with hook:
        video, latents = pipeline.inference(
            noise=noise,
            text_prompts=[str(task["prompt"])],
            return_latents=True,
            initial_latent=None,
            viewmats=stale_viewmats,
            Ks=intrinsics,
        )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    pixels = video[0].permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
    write_video(str(video_path), pixels, fps=int(task["fps"]))
    pipeline.vae.model.clear_cache()

    metadata = {
        **task,
        "status": "completed",
        "video_path": str(video_path),
        "minwm_revision": minwm_revision,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "model_config_path": str(model_config_path.resolve()),
        "experiment_config_path": str(experiment_config_path.resolve()),
        "interruption_strategy": "condition_swap",
        # Condition swapping reuses the in-progress latent and does not
        # explicitly discard completed generator calls. Restart baselines can
        # overwrite this field in later phases.
        "discarded_compute_ms": 0.0,
        "wall_seconds_including_decode": wall_seconds,
        "first_chunk_seconds_excluding_decode": getattr(pipeline, "last_chunk0_latency", None),
        "peak_cuda_memory_gb": peak_memory_gb,
        "low_memory_mode": bool(getattr(pipeline, "reactive_low_memory", False)),
        "prompt_cache_stats": dict(getattr(pipeline, "reactive_prompt_cache_stats", {})),
        "latent_shape": list(latents.shape),
        "pixel_frame_count": int(video.shape[1]),
        "generator_calls": hook.events,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
