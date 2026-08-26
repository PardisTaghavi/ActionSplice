"""Generate official HY-World 1.5 full-rollback and wait examples."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..backends.conditioning import (
    build_event_stale_command_schedules,
    commands_to_hyworld15_actions,
    commands_to_viewmats,
    expand_chunk_action_blocks,
    make_intrinsics,
)
from ..backends.hyworld15 import HYWorldPlayCondition
from ..backends.hyworld15_baselines import (
    BASELINE_POLICIES,
    OfficialHYWorldPlayBaselineHook,
)
from ..data.manifest import load_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyworld-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(BASELINE_POLICIES), required=True)
    parser.add_argument("--interruptions", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--receipt-step", type=int, choices=(1, 2, 3))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--attn-mode",
        choices=("torch", "flash"),
        default=os.environ.get("HYWORLD_ATTN_MODE", "torch"),
    )
    parser.add_argument("--offloading", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_baseline_task(
    config_path: Path,
    *,
    interruptions: int,
    prompt_index: int = 0,
    seed_override: int | None = None,
    receipt_override: int | None = None,
) -> dict[str, Any]:
    """Load and validate one interruption-count example."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    chunk_size = int(config.get("chunk_size", 4))
    denoising_steps = int(config.get("denoising_steps", 4))
    if chunk_size != 4 or denoising_steps != 4:
        raise ValueError("Official HY baseline examples require chunk=4 and NFE=4")
    schedules = config["command_block_schedules_by_interruptions"]
    key = str(interruptions)
    if key not in schedules:
        raise ValueError(f"No schedule configured for {interruptions} interruptions")
    blocks = [str(value) for value in schedules[key]]
    changed_blocks = [
        index for index in range(1, len(blocks)) if blocks[index] != blocks[index - 1]
    ]
    if len(changed_blocks) != interruptions:
        raise ValueError(
            f"Schedule {key} has {len(changed_blocks)} changes, expected {interruptions}"
        )
    if any(right - left < 2 for left, right in zip(changed_blocks, changed_blocks[1:])):
        raise ValueError("Wait examples require one full response block between events")
    if changed_blocks[-1] >= len(blocks) - 1:
        raise ValueError("Schedule needs a response block after its final event")

    prompts = load_prompts(config_path, str(config["prompt_file"]))
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise ValueError(f"prompt-index must be in [0, {len(prompts) - 1}]")
    seeds = [int(value) for value in config.get("seeds", [0])]
    if not seeds and seed_override is None:
        raise ValueError("Config must provide at least one seed")
    seed = int(seed_override if seed_override is not None else seeds[0])
    receipt = int(
        receipt_override if receipt_override is not None else config.get("runtime_receipt_step", 2)
    )
    if receipt not in (1, 2, 3):
        raise ValueError("runtime_receipt_step must be 1, 2, or 3")

    requested_commands = expand_chunk_action_blocks(blocks, chunk_size)
    stale_all = build_event_stale_command_schedules(blocks, chunk_size)
    event_pose_indices = [index * chunk_size for index in changed_blocks]
    stale_commands_by_event = {event: stale_all[event] for event in event_pose_indices}
    wait_blocks = list(blocks)
    for block_index in changed_blocks:
        wait_blocks[block_index] = blocks[block_index - 1]
    wait_commands = expand_chunk_action_blocks(wait_blocks, chunk_size)
    return {
        "prompt": prompts[prompt_index],
        "prompt_index": prompt_index,
        "seed": seed,
        "receipt_step": receipt,
        "interruptions": interruptions,
        "chunk_size": chunk_size,
        "denoising_steps": denoising_steps,
        "command_blocks": blocks,
        "wait_command_blocks": wait_blocks,
        "requested_commands": requested_commands,
        "wait_commands": wait_commands,
        "stale_commands_by_event": stale_commands_by_event,
        "event_pose_indices": event_pose_indices,
        "num_latent_frames": chunk_size * len(blocks),
    }


def main() -> None:
    args = build_parser().parse_args()
    import torch

    job_started = time.perf_counter()
    root = args.hyworld_root.resolve()
    model_path = args.model_path.resolve()
    action_checkpoint = args.action_checkpoint.resolve()
    reference_image = args.reference_image.resolve()
    config_path = args.experiment_config.resolve()
    for required in (
        root,
        model_path,
        action_checkpoint,
        reference_image,
        config_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    task = load_baseline_task(
        config_path,
        interruptions=args.interruptions,
        prompt_index=args.prompt_index,
        seed_override=args.seed,
        receipt_override=args.receipt_step,
    )
    run_id = (
        f"hyworld15_{args.policy}_i{args.interruptions}_r{task['receipt_step']}_s{task['seed']}"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{run_id}.mp4"
    metrics_path = output_dir / f"{run_id}.json"
    if video_path.exists() and metrics_path.exists() and not args.overwrite:
        print(f"skipped existing run: {run_id}", flush=True)
        return

    from hyvideo.commons import maybe_fallback_attn_mode
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    selected_attn_backend = maybe_fallback_attn_mode(args.attn_mode)
    print(f"selected_attn_backend={selected_attn_backend}", flush=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError("This baseline job is intentionally configured for one H100")
    initialize_parallel_state(sp=1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    initialize_infer_state(
        SimpleNamespace(
            sage_blocks_range="0-0",
            use_sageattn=False,
            enable_torch_compile=False,
            use_fp8_gemm=False,
            quant_type="fp8-per-block",
            include_patterns="double_blocks",
            use_vae_parallel=False,
        )
    )

    _print_gpu(torch, device)
    model_started = time.perf_counter()
    pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=str(model_path),
        transformer_version="480p_i2v",
        enable_offloading=args.offloading,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=str(action_checkpoint),
    )
    pipeline.transformer.set_attn_mode(selected_attn_backend)
    pipeline.transformer.config.attn_mode = selected_attn_backend
    torch.cuda.synchronize(device)
    model_load_wall_ms = (time.perf_counter() - model_started) * 1000.0
    torch.cuda.reset_peak_memory_stats(device)

    requested = _condition_from_commands(task["requested_commands"])
    wait_condition = _condition_from_commands(task["wait_commands"])
    stale_by_event = {
        int(event): _condition_from_commands(commands)
        for event, commands in task["stale_commands_by_event"].items()
    }
    hook = OfficialHYWorldPlayBaselineHook(
        pipeline=pipeline,
        wait_condition=wait_condition,
        stale_conditions_by_event=stale_by_event,
        event_pose_indices=task["event_pose_indices"],
        runtime_receipt_steps=[task["receipt_step"]],
        policy=args.policy,
    )
    video_length = (int(task["num_latent_frames"]) - 1) * 4 + 1
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with hook:
        latent_output = pipeline(
            prompt=str(task["prompt"]),
            negative_prompt="",
            reference_image=str(reference_image),
            aspect_ratio=f"{args.width}:{args.height}",
            num_inference_steps=4,
            video_length=video_length,
            seed=int(task["seed"]),
            output_type="latent",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            viewmats=requested.viewmats,
            Ks=requested.intrinsics,
            action=requested.actions,
            few_step=True,
            chunk_latent_frames=4,
            model_type="ar",
            user_height=args.height,
            user_width=args.width,
            enable_sr=False,
            transformer_resident_ar_rollout=True,
        )
    torch.cuda.synchronize(device)
    latent_inference_wall_ms = (time.perf_counter() - inference_started) * 1000.0
    if hook.result is None:
        raise RuntimeError("HY-World baseline hook did not receive ar_rollout")
    if int(hook.result.metrics["interruption_count"]) != args.interruptions:
        raise RuntimeError("Executed interruption count differs from requested count")

    latents = latent_output.videos
    decode_started = time.perf_counter()
    video = _decode_latents(pipeline, latents)
    torch.cuda.synchronize(device)
    decode_wall_ms = (time.perf_counter() - decode_started) * 1000.0
    save_started = time.perf_counter()
    _save_video(video, video_path, fps=args.fps)
    video_save_wall_ms = (time.perf_counter() - save_started) * 1000.0

    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    rgb_frames = int(video.shape[2])
    metrics = {
        "run_id": run_id,
        "status": "completed",
        "policy": args.policy,
        "interruption_count": args.interruptions,
        "receipt_step": task["receipt_step"],
        "prompt": task["prompt"],
        "prompt_index": task["prompt_index"],
        "seed": task["seed"],
        "command_blocks": task["command_blocks"],
        "wait_command_blocks": task["wait_command_blocks"],
        "event_pose_indices": task["event_pose_indices"],
        "reference_image": str(reference_image),
        "video_path": str(video_path),
        "rgb_frame_count": rgb_frames,
        "fps": args.fps,
        "video_duration_s": rgb_frames / args.fps,
        "height": args.height,
        "width": args.width,
        "hyworld_revision": _git_revision(root),
        "model_path": str(model_path),
        "action_checkpoint": str(action_checkpoint),
        "offloading": args.offloading,
        "selected_attn_backend": selected_attn_backend,
        "model_load_wall_ms": model_load_wall_ms,
        "latent_inference_wall_ms": latent_inference_wall_ms,
        "pipeline_prepost_wall_ms": max(
            0.0,
            latent_inference_wall_ms - hook.result.metrics["rollout_wall_ms"],
        ),
        "decode_wall_ms": decode_wall_ms,
        "video_save_wall_ms": video_save_wall_ms,
        "job_wall_ms": (time.perf_counter() - job_started) * 1000.0,
        "peak_cuda_allocated_gib": peak_allocated_gib,
        "peak_cuda_reserved_gib": peak_reserved_gib,
        "max_host_rss_gib": _max_rss_gib(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        **hook.result.metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _print_summary(metrics, metrics_path)


def _condition_from_commands(commands: list[str]) -> HYWorldPlayCondition:
    import torch

    frame_count = len(commands) + 1
    return HYWorldPlayCondition(
        viewmats=torch.from_numpy(commands_to_viewmats(commands)).unsqueeze(0),
        intrinsics=torch.from_numpy(
            make_intrinsics(frame_count, fx=0.5050505, fy=0.89786756)
        ).unsqueeze(0),
        actions=torch.from_numpy(commands_to_hyworld15_actions(commands)).unsqueeze(0),
    )


def _decode_latents(pipeline: Any, latents: Any) -> Any:
    import torch
    from hyvideo.commons import auto_offload_model

    # AR inference leaves large modules and text/vision KV tensors resident.
    # Evict everything unrelated to decoding before placing the VAE and
    # latents on the GPU.  Manual decode must also run in inference mode:
    # otherwise trainable VAE parameters retain an autograd graph for every
    # decoded frame and exhaust an 80 GB H100.
    latents = latents.detach().to("cpu")
    for name in ("transformer", "text_encoder", "text_encoder_2", "byt5_model", "vision_encoder"):
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
    print(
        "pre_decode_cuda_allocated_gib="
        f"{torch.cuda.memory_allocated(pipeline.execution_device) / 2**30:.2f} "
        "vae_spatial_tiling=false inference_mode=true",
        flush=True,
    )

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


def _save_video(video: Any, path: Path, *, fps: int) -> None:
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


def _print_gpu(torch: Any, device: Any) -> None:
    properties = torch.cuda.get_device_properties(device)
    print(
        f"GPU={properties.name} total_vram_gib={properties.total_memory / 2**30:.2f}",
        flush=True,
    )


def _print_summary(metrics: dict[str, Any], metrics_path: Path) -> None:
    print(
        " ".join(
            [
                f"completed={metrics['run_id']}",
                f"generator_calls={metrics['generator_call_count']}",
                f"discarded_nfe={metrics['discarded_nfe']}",
                f"rollout_s={metrics['rollout_wall_ms'] / 1000:.3f}",
                f"decode_s={metrics['decode_wall_ms'] / 1000:.3f}",
                f"peak_allocated_gib={metrics['peak_cuda_allocated_gib']:.2f}",
                f"peak_reserved_gib={metrics['peak_cuda_reserved_gib']:.2f}",
            ]
        ),
        flush=True,
    )
    for event in metrics["interruption_events"]:
        print(
            " ".join(
                [
                    f"event={event['event_index']}",
                    f"pose={event['event_pose_index']}",
                    f"receipt={event['receipt_step']}",
                    f"discarded_nfe={event['discarded_nfe']}",
                    f"response_latency_ms={event['response_latency_ms']:.3f}",
                ]
            ),
            flush=True,
        )
    print(f"metrics={metrics_path}", flush=True)


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / 2**30
    return value * 1024.0 / 2**30


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
