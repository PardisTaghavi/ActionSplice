"""Build unique counterfactual-transport capture tasks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..backends.conditioning import SCENARIOS, build_command_pair
from .manifest import _slug, load_config, load_config_prompts, select_shard


def build_transport_capture_tasks(config_path: Path) -> list[dict[str, Any]]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    prompts = load_config_prompts(config_path, config)
    num_frames = int(config["num_latent_frames"])
    chunk_size = int(config["chunk_size"])
    event_pose_index = int(config["event_pose_index"])
    multiple_offset_config = "intra_chunk_offsets" in config
    if multiple_offset_config:
        intra_chunk_offsets = tuple(int(value) for value in config["intra_chunk_offsets"])
        if "intra_chunk_offset" in config:
            raise ValueError("Specify either intra_chunk_offset or intra_chunk_offsets, not both")
    else:
        intra_chunk_offsets = (int(config.get("intra_chunk_offset", 0)),)
    if not intra_chunk_offsets or len(set(intra_chunk_offsets)) != len(intra_chunk_offsets):
        raise ValueError("intra_chunk_offsets must be nonempty and unique")
    denoising_steps = int(config["denoising_steps"])
    scenario_assignment = str(config.get("scenario_assignment", "cross_product"))
    if scenario_assignment not in {"cross_product", "round_robin"}:
        raise ValueError("scenario_assignment must be cross_product or round_robin")
    prompt_seed_stride = int(config.get("prompt_seed_stride", 0))
    if prompt_seed_stride < 0:
        raise ValueError("prompt_seed_stride must be nonnegative")
    train_prompt_count = int(config.get("train_prompt_count", len(prompts)))
    if not 0 < train_prompt_count <= len(prompts):
        raise ValueError("train_prompt_count must select a nonempty prompt prefix")
    if denoising_steps < 2:
        raise ValueError("denoising_steps must be at least two")
    if event_pose_index % chunk_size:
        raise ValueError("event_pose_index must be a chunk boundary")
    for intra_chunk_offset in intra_chunk_offsets:
        if intra_chunk_offset and not 0 < intra_chunk_offset < chunk_size:
            raise ValueError("Every intra_chunk_offset must select a nonempty chunk suffix")

    tasks: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_key = f"p{prompt_index:02d}-{_slug(prompt)}"
        for seed in config["seeds"]:
            effective_seed = int(seed) + prompt_index * prompt_seed_stride
            for intra_chunk_offset in intra_chunk_offsets:
                effective_pose_index = event_pose_index + intra_chunk_offset
                scenario_names = list(config["scenarios"])
                if scenario_assignment == "round_robin":
                    scenario_names = [scenario_names[prompt_index % len(scenario_names)]]
                for scenario_name in scenario_names:
                    scenario = SCENARIOS[scenario_name]
                    if scenario.is_control:
                        raise ValueError(
                            "Transport capture requires a condition change; "
                            f"remove {scenario_name!r}"
                        )
                    stale_commands, requested_commands = build_command_pair(
                        scenario_name,
                        num_frames,
                        effective_pose_index,
                    )
                    key = "|".join(
                        [
                            str(prompt_index),
                            str(effective_seed),
                            scenario_name,
                            str(event_pose_index),
                            str(intra_chunk_offset),
                        ]
                    )
                    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
                    offset_tag = f"__m{intra_chunk_offset}" if multiple_offset_config else ""
                    capture_id = (
                        f"{prompt_key}__s{effective_seed:03d}__{scenario_name}"
                        f"{offset_tag}__transport__{digest}"
                    )
                    tasks.append(
                        {
                            "capture_id": capture_id,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "master_id": f"cst-paper-p{prompt_index:03d}",
                            "dataset_split": (
                                "train" if prompt_index < train_prompt_count else "heldout"
                            ),
                            "seed": effective_seed,
                            "prompt_seed_stride": prompt_seed_stride,
                            "scenario_assignment": scenario_assignment,
                            "scenario": scenario_name,
                            "scenario_description": scenario.description,
                            "analysis_scope": ("stress" if scenario.is_stress else "primary"),
                            "old_action": scenario.old_action,
                            "new_action": scenario.new_action,
                            "event_pose_index": event_pose_index,
                            "effective_pose_index": effective_pose_index,
                            "intra_chunk_offset": intra_chunk_offset,
                            "event_chunk_index": event_pose_index // chunk_size,
                            "num_latent_frames": num_frames,
                            "chunk_size": chunk_size,
                            "denoising_steps": denoising_steps,
                            "receipt_steps": list(range(1, denoising_steps)),
                            "stale_commands": stale_commands,
                            "requested_commands": requested_commands,
                        }
                    )
    return tasks


__all__ = ["build_transport_capture_tasks", "select_shard"]
