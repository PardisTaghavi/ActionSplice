"""Generate final recurrent on-policy minWM CST-R captures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..backends.conditioning import commands_to_viewmats, make_intrinsics
from ..data.capture import save_transport_capture
from ..data.recurrent_manifest import build_recurrent_capture_tasks, select_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--transport-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_under(root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    args = build_parser().parse_args()
    import torch

    from ..backends import get_backend
    from ..backends.minwm_recurrent import capture_recurrent_cst_r
    from ..backends.minwm_wan import load_wan_pipeline, minwm_commit
    from ..core.runtime import load_transport_model

    minwm_root = args.minwm_root.resolve()
    transport_checkpoint = args.transport_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    capture_dir = output_dir / "captures"
    metadata_dir = output_dir / "metadata"
    sequence_dir = output_dir / "sequences"
    trajectory_dir = output_dir / "trajectories"
    for directory in (capture_dir, metadata_dir, sequence_dir, trajectory_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = select_shard(
        build_recurrent_capture_tasks(args.experiment_config),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    pipeline, model_config, _ = load_wan_pipeline(
        minwm_root,
        _resolve_under(minwm_root, args.minwm_config).resolve(),
        _resolve_under(minwm_root, args.checkpoint).resolve(),
        low_memory=args.low_memory,
    )
    from wan_utils.misc import set_seed

    device = next(pipeline.generator.parameters()).device
    dtype = torch.bfloat16
    corrector, checkpoint_metadata = load_transport_model(
        transport_checkpoint, device=device, dtype=dtype
    )
    get_backend("minwm").validate_checkpoint_config(
        checkpoint_metadata["model_config"], method="cst_r"
    )
    revision = minwm_commit(minwm_root)
    if tasks and int(model_config.num_frame_per_block) != int(tasks[0]["chunk_size"]):
        raise ValueError("Capture chunk size does not match minWM")

    for task_index, task in enumerate(tasks, start=1):
        sequence_id = str(task["recurrent_sequence_id"])
        summary_path = sequence_dir / f"{sequence_id}.json"
        expected_paths = [
            capture_dir / f"{sequence_id}__e{event_index:02d}.pt"
            for event_index in range(len(task["event_pose_indices"]))
        ]
        trajectory_path = trajectory_dir / f"{sequence_id}.pt"
        trajectory_ready = trajectory_path.exists() or not bool(
            task.get("save_trajectory_artifact", False)
        )
        if (
            summary_path.exists()
            and all(path.exists() for path in expected_paths)
            and trajectory_ready
            and not args.overwrite
        ):
            print(f"[{task_index}/{len(tasks)}] skipped {sequence_id}")
            continue

        print(f"[{task_index}/{len(tasks)}] {sequence_id}")
        set_seed(int(task["seed"]))
        frame_count = int(task["num_latent_frames"])
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
            torch.from_numpy(make_intrinsics(frame_count))
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        noise = torch.randn([1, frame_count, 16, 60, 104], device=device, dtype=dtype)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        captures, output, metrics = capture_recurrent_cst_r(
            pipeline=pipeline,
            noise=noise,
            text_prompt=str(task["prompt"]),
            stale_viewmats_by_event=stale_by_event,
            requested_viewmats=requested,
            intrinsics=intrinsics,
            event_pose_indices=task["event_pose_indices"],
            receipt_steps=task["runtime_receipt_steps"],
            corrector=corrector,
        )
        torch.cuda.synchronize()
        if len(captures) != len(expected_paths):
            raise RuntimeError("Capture count does not match the event schedule")

        common = {
            **task,
            "backbone": "minwm_wan_action2v",
            "native_state_layout": "BTCHW",
            "canonical_state_layout": "BTCHW",
            "stochastic_transitions": True,
            "minwm_revision": revision,
            "transport_checkpoint": checkpoint_metadata,
        }
        for event_index, (capture, path) in enumerate(zip(captures, expected_paths, strict=True)):
            capture.metadata.update(
                common,
                capture_id=path.stem,
                recurrent_event_index=event_index,
            )
            save_transport_capture(path, capture)
            (metadata_dir / f"{path.stem}.json").write_text(
                json.dumps(capture.metadata, indent=2) + "\n", encoding="utf-8"
            )

        summary = {
            **common,
            **metrics,
            "status": "completed",
            "capture_paths": [str(path) for path in expected_paths],
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        if bool(task.get("save_trajectory_artifact", False)):
            torch.save(
                {
                    "schema_version": 1,
                    "metadata": common,
                    "tensors": {
                        "initial_noise": noise.detach().cpu(),
                        "student_output": output.detach().cpu(),
                        "requested_viewmats": requested.detach().cpu(),
                        "intrinsics": intrinsics.detach().cpu(),
                    },
                },
                trajectory_path,
            )
            summary["trajectory_artifact"] = str(trajectory_path)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        pipeline.vae.model.clear_cache()
        print(f"  completed captures={len(captures)}")


if __name__ == "__main__":
    main()
