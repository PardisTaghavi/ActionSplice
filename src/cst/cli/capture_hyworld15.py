"""Run recurrent CST capture on the official HY-World 1.5 checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ..backends.conditioning import (
    commands_to_hyworld15_actions,
    commands_to_viewmats,
    make_intrinsics,
)
from ..backends.hyworld15 import HYWorldPlayCondition, OfficialHYWorldPlayCSTHook
from ..data.capture import save_transport_capture
from ..data.recurrent_manifest import build_recurrent_capture_tasks, select_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyworld-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
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
    transport_checkpoint = (
        None if args.transport_checkpoint is None else args.transport_checkpoint.resolve()
    )
    reference_image = args.reference_image.resolve()
    for required in (
        root,
        model_path,
        action_checkpoint,
        reference_image,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
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
    from ..backends import get_backend
    from ..core.runtime import load_transport_model

    if transport_checkpoint is None:
        corrector = None
        corrector_metadata = None
    else:
        if not transport_checkpoint.exists():
            raise FileNotFoundError(transport_checkpoint)
        corrector, corrector_metadata = load_transport_model(
            transport_checkpoint,
            device=torch.device(f"cuda:{local_rank}"),
            dtype=torch.bfloat16,
        )
        get_backend("hyworld15").validate_checkpoint_config(
            corrector_metadata["model_config"], method="cst_r"
        )
    tasks = select_shard(
        build_recurrent_capture_tasks(args.experiment_config),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit_examples is not None:
        tasks = tasks[: args.limit_examples]
    output_dir = args.output_dir.resolve()
    capture_dir = output_dir / "captures"
    sequence_dir = output_dir / "sequences"
    capture_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    revision = _git_revision(root)

    for task in tasks:
        sequence_id = str(task["recurrent_sequence_id"])
        summary_path = sequence_dir / f"{sequence_id}.json"
        expected_paths = [
            capture_dir / f"{sequence_id}__e{index:02d}.pt"
            for index in range(len(task["event_pose_indices"]))
        ]
        if (
            summary_path.exists()
            and all(path.exists() for path in expected_paths)
            and not args.overwrite
        ):
            print(f"skipped {sequence_id}", flush=True)
            continue

        requested = _condition_from_commands(task["requested_commands"])
        stale_by_event = {
            int(event): _condition_from_commands(commands)
            for event, commands in task["stale_commands_by_event"].items()
        }
        hook = OfficialHYWorldPlayCSTHook(
            pipeline=pipeline,
            stale_conditions_by_event=stale_by_event,
            event_pose_indices=task["event_pose_indices"],
            runtime_receipt_steps=task["runtime_receipt_steps"],
            transport_model=corrector,
        )
        video_length = (int(task["num_latent_frames"]) - 1) * 4 + 1
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
            raise RuntimeError("HY-WorldPlay CST hook did not receive ar_rollout")
        if len(hook.result.captures) != len(expected_paths):
            raise RuntimeError("Recurrent capture count differs from event schedule")
        for event_index, (capture, path) in enumerate(
            zip(hook.result.captures, expected_paths, strict=True)
        ):
            capture.metadata.update(
                {
                    "capture_id": path.stem,
                    "recurrent_sequence_id": sequence_id,
                    "recurrent_event_index": event_index,
                    "prompt": task["prompt"],
                    "prompt_index": task["prompt_index"],
                    "seed": task["seed"],
                    "command_blocks": task["command_blocks"],
                    "requested_commands": task["requested_commands"],
                    "stale_commands": task["stale_commands_by_event"][
                        str(capture.metadata["event_pose_index"])
                        if str(capture.metadata["event_pose_index"])
                        in task["stale_commands_by_event"]
                        else int(capture.metadata["event_pose_index"])
                    ],
                    "hyworld_revision": revision,
                    "hyworld_model_path": str(model_path),
                    "hyworld_action_checkpoint": str(action_checkpoint),
                    "transport_checkpoint": corrector_metadata,
                }
            )
            save_transport_capture(path, capture)
        summary = {
            "recurrent_sequence_id": sequence_id,
            "status": "completed",
            "hyworld_revision": revision,
            "capture_paths": [str(path) for path in expected_paths],
            **hook.result.metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"completed {sequence_id}", flush=True)


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
