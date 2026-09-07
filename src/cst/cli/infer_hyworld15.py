"""Run one HY-WM1.5 CST-R or CST-T rollout and save its video and metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from ..data.inference import load_inference_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("cst_r", "cst_t"), required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--hyworld-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--attn-mode", choices=("torch", "flash"), default="torch")
    parser.add_argument("--offloading", action="store_true")
    parser.add_argument("--group-offloading", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    from ..backends import get_backend
    from ..backends.hyworld15_inference import OfficialHYWorldPlayInferenceHook
    from ..cli._hyworld15_io import (
        condition_from_commands,
        decode_latents,
        git_revision,
        save_video,
    )
    from ..core.runtime import load_transport_model

    task = load_inference_task(args.task_config, method=args.method)
    root = args.hyworld_root.resolve()
    model_path = args.model_path.resolve()
    action_checkpoint = args.action_checkpoint.resolve()
    reference_image = args.reference_image.resolve()
    for required in (root, model_path, action_checkpoint, reference_image):
        if not required.exists():
            raise FileNotFoundError(required)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from hyvideo.commons import is_flash2_available, is_flash3_available
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("HY-WM1.5 ActionSplice inference uses one GPU")
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
    if args.attn_mode == "torch":
        selected_attention = "torch"
    elif is_flash3_available():
        selected_attention = "flash3"
    elif is_flash2_available():
        selected_attention = "flash2"
    else:
        raise RuntimeError("Flash attention was requested but is not installed")
    pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=str(model_path),
        transformer_version="480p_i2v",
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=str(action_checkpoint),
    )
    pipeline.transformer.set_attn_mode(selected_attention)
    pipeline.transformer.config.attn_mode = selected_attention
    corrector, checkpoint_metadata = load_transport_model(
        args.transport_checkpoint, device=device, dtype=torch.bfloat16
    )
    get_backend("hyworld15").validate_checkpoint_config(
        checkpoint_metadata["model_config"], method=args.method
    )

    requested = condition_from_commands(task["requested_commands"])
    stale_by_event = {
        int(event): condition_from_commands(commands)
        for event, commands in task["stale_commands_by_event"].items()
    }
    hook = OfficialHYWorldPlayInferenceHook(
        pipeline=pipeline,
        stale_conditions_by_event=stale_by_event,
        event_pose_indices=task["event_pose_indices"],
        receipt_steps=task["receipt_steps"],
        corrector=corrector,
        intra_chunk_offsets_by_event=(
            task["intra_chunk_offsets_by_event"] if args.method == "cst_t" else None
        ),
    )
    video_length = (int(task["num_latent_frames"]) - 1) * 4 + 1
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
    if hook.result is None:
        raise RuntimeError("HY-WorldPlay did not call the ActionSplice AR hook")
    video = decode_latents(pipeline, latent_output.videos)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, output, fps=args.fps)
    record = {
        "status": "completed",
        "backend": "hyworld15",
        "method": args.method,
        "task": task,
        "reference_image": str(reference_image),
        "video_path": str(output),
        "hyworld_revision": git_revision(root),
        "checkpoint": checkpoint_metadata,
        "runtime": hook.result.metrics,
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"video={output}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
