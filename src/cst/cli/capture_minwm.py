"""Generate paired old/new sampler-state captures on minWM Wan."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..backends.conditioning import commands_to_viewmats, make_intrinsics
from ..backends.minwm_capture import capture_transport_pair, save_transport_capture
from ..data.transport_manifest import build_transport_capture_tasks, select_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--minwm-root", type=Path, required=True)
    parser.add_argument(
        "--minwm-config",
        type=Path,
        default=Path("Wan21/configs/causal_forcing_dmd_camera.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ckpts/Wan21/Action2V/dmd/model.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trace-cache-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _resolve_under(root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    args = build_parser().parse_args()
    import torch

    from ..backends.minwm_wan import load_wan_pipeline, minwm_commit

    minwm_root = args.minwm_root.resolve()
    output_dir = args.output_dir.resolve()
    capture_dir = output_dir / "captures"
    metadata_dir = output_dir / "metadata"
    capture_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tasks = select_shard(
        build_transport_capture_tasks(args.experiment_config),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    config_path = _resolve_under(minwm_root, args.minwm_config).resolve()
    checkpoint_path = _resolve_under(minwm_root, args.checkpoint).resolve()
    pipeline, model_config, _ = load_wan_pipeline(
        minwm_root,
        config_path,
        checkpoint_path,
        low_memory=args.low_memory,
    )
    from wan_utils.misc import set_seed

    revision = minwm_commit(minwm_root)
    if tasks and int(model_config.num_frame_per_block) != int(tasks[0]["chunk_size"]):
        raise ValueError("Configured capture chunk size does not match minWM")

    for index, task in enumerate(tasks, start=1):
        capture_id = str(task["capture_id"])
        capture_path = capture_dir / f"{capture_id}.pt"
        metadata_path = metadata_dir / f"{capture_id}.json"
        if capture_path.exists() and metadata_path.exists() and not args.overwrite:
            print(f"[{index}/{len(tasks)}] skipped {capture_id}")
            continue

        print(f"[{index}/{len(tasks)}] {capture_id}")
        set_seed(int(task["seed"]))
        device = next(pipeline.generator.parameters()).device
        dtype = torch.bfloat16
        num_frames = int(task["num_latent_frames"])
        stale_viewmats = (
            torch.from_numpy(commands_to_viewmats(task["stale_commands"]))
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        requested_viewmats = (
            torch.from_numpy(commands_to_viewmats(task["requested_commands"]))
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        intrinsics = (
            torch.from_numpy(make_intrinsics(num_frames))
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        noise = torch.randn(
            [1, num_frames, 16, 60, 104],
            device=device,
            dtype=dtype,
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        capture = capture_transport_pair(
            pipeline=pipeline,
            noise=noise,
            text_prompt=str(task["prompt"]),
            stale_viewmats=stale_viewmats,
            requested_viewmats=requested_viewmats,
            intrinsics=intrinsics,
            event_pose_index=int(task["event_pose_index"]),
            intra_chunk_offset=int(task.get("intra_chunk_offset", 0)),
            trace_cache_indices=args.trace_cache_indices,
        )
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        capture.metadata.update(
            {
                **task,
                "minwm_revision": revision,
                "backbone": "minwm_wan_action2v",
                "native_state_layout": "BTCHW",
                "canonical_state_layout": "BTCHW",
                "stochastic_transitions": True,
                "wall_seconds": wall_seconds,
                "peak_cuda_memory_gb": peak_gb,
            }
        )
        save_transport_capture(capture_path, capture)
        metadata_path.write_text(
            json.dumps(capture.metadata, indent=2),
            encoding="utf-8",
        )
        pipeline.vae.model.clear_cache()
        print(f"  completed wall={wall_seconds:.2f}s peak={peak_gb:.2f} GiB")


if __name__ == "__main__":
    main()
