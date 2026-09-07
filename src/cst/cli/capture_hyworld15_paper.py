"""Capture independent HY-WM1.5 CST-R or CST-T datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..backends.conditioning import (
    commands_to_hyworld15_actions,
    commands_to_viewmats,
    make_intrinsics,
)
from ..backends.hyworld15 import HYWorldPlayCondition, OfficialHYWorldPlayCSTHook
from ..data.capture import save_transport_capture
from ..data.manifest import select_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cst_r", "cst_t"), required=True)
    parser.add_argument("--hyworld-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--offloading", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    root = args.hyworld_root.resolve()
    model_path = args.model_path.resolve()
    action_checkpoint = args.action_checkpoint.resolve()
    task_manifest = args.task_manifest.resolve()
    transport_checkpoint = (
        None if args.transport_checkpoint is None else args.transport_checkpoint.resolve()
    )
    if args.mode == "cst_r" and transport_checkpoint is None:
        raise ValueError("CST-R capture requires a frozen HY CST-R checkpoint")
    if args.mode == "cst_t" and transport_checkpoint is not None:
        raise ValueError("CST-T capture is independent and uses no CST-R checkpoint")
    for required in (root, model_path, action_checkpoint, task_manifest):
        if not required.exists():
            raise FileNotFoundError(required)
    if transport_checkpoint is not None and not transport_checkpoint.is_file():
        raise FileNotFoundError(transport_checkpoint)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError("HY paper capture shards across one-GPU jobs")
    initialize_parallel_state(sp=1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
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

    corrector = None
    corrector_metadata = None
    if transport_checkpoint is not None:
        from ..core.runtime import load_transport_model

        corrector, corrector_metadata = load_transport_model(
            transport_checkpoint,
            device=torch.device(f"cuda:{local_rank}"),
            dtype=torch.bfloat16,
        )
        config = corrector.config
        if str(config.transport_role) != "action_h0_state":
            raise ValueError("CST-R requires an action_h0_state checkpoint")
        if int(config.latent_channels) != 32:
            raise ValueError("CST-R checkpoint must use 32 HY latent channels")
        if str(config.target_parameterization) != "state":
            raise ValueError("HY checkpoint must predict deterministic sampler state")

    tasks = json.loads(task_manifest.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Task manifest must contain a JSON list")
    expected_total = 150 if args.mode == "cst_r" else 450
    if len(tasks) != expected_total:
        raise ValueError(
            f"{args.mode} manifest must contain {expected_total} tasks, got {len(tasks)}"
        )
    tasks = select_shard(
        tasks,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit_examples is not None:
        tasks = tasks[: args.limit_examples]

    output_dir = args.output_dir.resolve()
    capture_dir = output_dir / "captures"
    metadata_dir = output_dir / "metadata"
    sequence_dir = output_dir / "sequences"
    capture_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "cst_r":
        sequence_dir.mkdir(parents=True, exist_ok=True)
    revision = _git_revision(root)
    verified_image_hashes: dict[Path, str] = {}

    for task in tasks:
        reference_image = Path(str(task["reference_image"])).resolve()
        if not reference_image.is_file():
            raise FileNotFoundError(reference_image)
        actual_hash = verified_image_hashes.get(reference_image)
        if actual_hash is None:
            actual_hash = _sha256(reference_image)
            verified_image_hashes[reference_image] = actual_hash
        if actual_hash != str(task["reference_image_sha256"]):
            raise ValueError(f"Reference image changed after manifest export: {reference_image}")
        if str(task.get("backbone")) != "official_hyworld15_action2v":
            raise ValueError("Task manifest has the wrong backbone")
        if args.mode == "cst_r":
            expected_paths, summary_path = _correct_output_paths(
                task, capture_dir=capture_dir, sequence_dir=sequence_dir
            )
        else:
            capture_id = str(task["capture_id"])
            expected_paths = [capture_dir / f"{capture_id}.pt"]
            summary_path = metadata_dir / f"{capture_id}.json"
        if (
            summary_path.exists()
            and all(path.exists() and path.stat().st_size > 0 for path in expected_paths)
            and not args.overwrite
        ):
            print(f"skipped {summary_path.stem}", flush=True)
            continue

        requested = _condition_from_commands(task["requested_commands"])
        if args.mode == "cst_r":
            event_pose_indices = [int(value) for value in task["event_pose_indices"]]
            stale_by_event = {
                event: _condition_from_commands(
                    _event_value(task["stale_commands_by_event"], event)
                )
                for event in event_pose_indices
            }
            receipt_steps = [int(value) for value in task["runtime_receipt_steps"]]
            offsets: dict[int, int] = {}
        else:
            event = int(task["event_pose_index"])
            event_pose_indices = [event]
            stale_by_event = {event: _condition_from_commands(task["stale_commands"])}
            receipt_steps = [1]
            offsets = {event: int(task["intra_chunk_offset"])}

        hook = OfficialHYWorldPlayCSTHook(
            pipeline=pipeline,
            stale_conditions_by_event=stale_by_event,
            event_pose_indices=event_pose_indices,
            runtime_receipt_steps=receipt_steps,
            transport_model=corrector,
            intra_chunk_offsets_by_event=offsets,
        )
        latent_frames = int(task["num_latent_frames"])
        video_length = (latent_frames - 1) * 4 + 1
        torch.cuda.reset_peak_memory_stats()
        with hook:
            pipeline(
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
        if hook.result is None:
            raise RuntimeError("HY-WorldPlay hook did not receive ar_rollout")
        if len(hook.result.captures) != len(expected_paths):
            raise RuntimeError("HY capture count differs from the task manifest")

        for event_index, (capture, path) in enumerate(
            zip(hook.result.captures, expected_paths, strict=True)
        ):
            common = {
                "capture_id": path.stem,
                "dataset_name": (
                    "cst_r_paper_150_v1" if args.mode == "cst_r" else "cst_t_paper_150_v1"
                ),
                "dataset_split": str(task["dataset_split"]),
                "master_id": str(task["master_id"]),
                "scene_id": str(task["scene_id"]),
                "prompt": str(task["prompt"]),
                "prompt_index": int(task["prompt_index"]),
                "reference_image": str(reference_image),
                "reference_image_sha256": str(task["reference_image_sha256"]),
                "seed": int(task["seed"]),
                "hyworld_revision": revision,
                "hyworld_model_path": str(model_path),
                "hyworld_action_checkpoint": str(action_checkpoint),
                "transport_checkpoint": corrector_metadata,
            }
            if args.mode == "cst_r":
                common.update(
                    recurrent_sequence_id=str(task["recurrent_sequence_id"]),
                    recurrent_event_index=event_index,
                    command_blocks=list(task["command_blocks"]),
                    requested_commands=list(task["requested_commands"]),
                    stale_commands=list(
                        _event_value(
                            task["stale_commands_by_event"],
                            int(capture.metadata["event_pose_index"]),
                        )
                    ),
                )
            else:
                common.update(
                    scenario=str(task["scenario"]),
                    old_action=str(task["old_action"]),
                    new_action=str(task["new_action"]),
                    requested_commands=list(task["requested_commands"]),
                    stale_commands=list(task["stale_commands"]),
                    intra_chunk_offset=int(task["intra_chunk_offset"]),
                )
            capture.metadata.update(common)
            save_transport_capture(path, capture)

        summary = {
            "status": "completed",
            "mode": args.mode,
            "master_id": str(task["master_id"]),
            "capture_paths": [str(path) for path in expected_paths],
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            **hook.result.metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"completed {summary_path.stem}", flush=True)
        torch.cuda.empty_cache()

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _condition_from_commands(commands: list[str]) -> HYWorldPlayCondition:
    import torch

    commands = [str(value) for value in commands]
    frame_count = len(commands) + 1
    return HYWorldPlayCondition(
        viewmats=torch.from_numpy(commands_to_viewmats(commands)).unsqueeze(0),
        intrinsics=torch.from_numpy(
            make_intrinsics(frame_count, fx=0.5050505, fy=0.89786756)
        ).unsqueeze(0),
        actions=torch.from_numpy(commands_to_hyworld15_actions(commands)).unsqueeze(0),
    )


def _event_value(mapping: dict[Any, Any], event: int) -> Any:
    if event in mapping:
        return mapping[event]
    if str(event) in mapping:
        return mapping[str(event)]
    raise KeyError(event)


def _correct_output_paths(
    task: dict[str, Any], *, capture_dir: Path, sequence_dir: Path
) -> tuple[list[Path], Path]:
    sequence_id = str(task["recurrent_sequence_id"])
    paths = [
        capture_dir / f"{sequence_id}__e{index:02d}.pt"
        for index in range(len(task["event_pose_indices"]))
    ]
    return paths, sequence_dir / f"{sequence_id}.json"


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
