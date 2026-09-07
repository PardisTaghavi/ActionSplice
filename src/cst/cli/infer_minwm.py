"""Run one minWM CST-R or CST-T rollout and save its video and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..backends.conditioning import commands_to_viewmats, make_intrinsics
from ..data.inference import load_inference_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("cst_r", "cst_t"), required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--minwm-root", type=Path, required=True)
    parser.add_argument(
        "--minwm-config",
        type=Path,
        default=Path("Wan21/configs/causal_forcing_dmd_camera.yaml"),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=Path("ckpts/Wan21/Action2V/dmd/model.pt"),
    )
    parser.add_argument("--transport-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--low-memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _resolve_under(root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from torchvision.io import write_video

    from ..backends import get_backend
    from ..backends.minwm_inference import run_minwm_cst
    from ..backends.minwm_wan import load_wan_pipeline, minwm_commit
    from ..core.runtime import load_transport_model

    task = load_inference_task(args.task_config, method=args.method)
    root = args.minwm_root.resolve()
    transport_checkpoint = args.transport_checkpoint.resolve()
    output = args.output.resolve()
    pipeline, model_config, _ = load_wan_pipeline(
        root,
        _resolve_under(root, args.minwm_config).resolve(),
        _resolve_under(root, args.backbone_checkpoint).resolve(),
        low_memory=args.low_memory,
    )
    if int(model_config.num_frame_per_block) != int(task["chunk_size"]):
        raise ValueError("Task chunk size does not match minWM")
    from wan_utils.misc import set_seed

    set_seed(int(task["seed"]))
    device = next(pipeline.generator.parameters()).device
    dtype = torch.bfloat16
    corrector, checkpoint_metadata = load_transport_model(
        transport_checkpoint, device=device, dtype=dtype
    )
    get_backend("minwm").validate_checkpoint_config(
        checkpoint_metadata["model_config"], method=args.method
    )
    requested = (
        torch.from_numpy(commands_to_viewmats(task["requested_commands"]))
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    stale_by_event = {
        int(event): torch.from_numpy(commands_to_viewmats(commands))
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        for event, commands in task["stale_commands_by_event"].items()
    }
    intrinsics = (
        torch.from_numpy(make_intrinsics(int(task["num_latent_frames"])))
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    noise = torch.randn(
        [1, int(task["num_latent_frames"]), 16, 60, 104],
        device=device,
        dtype=dtype,
    )
    result = run_minwm_cst(
        pipeline=pipeline,
        noise=noise,
        text_prompt=str(task["prompt"]),
        requested_viewmats=requested,
        stale_viewmats_by_event=stale_by_event,
        intrinsics=intrinsics,
        event_pose_indices=task["event_pose_indices"],
        receipt_steps=task["receipt_steps"],
        corrector=corrector,
        intra_chunk_offsets_by_event=(
            task["intra_chunk_offsets_by_event"] if args.method == "cst_t" else None
        ),
    )
    if result.video is None:
        raise RuntimeError("minWM inference returned no decoded video")
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        result.video[0].permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
    )
    write_video(str(output), pixels, fps=args.fps)
    record = {
        "status": "completed",
        "backend": "minwm",
        "method": args.method,
        "task": task,
        "video_path": str(output),
        "minwm_revision": minwm_commit(root),
        "checkpoint": checkpoint_metadata,
        "runtime": result.metrics,
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"video={output}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
