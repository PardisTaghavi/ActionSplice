"""Build long student-rollout tasks for recurrent CST capture."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from ..backends.conditioning import (
    VALID_ACTIONS,
    build_event_stale_command_schedules,
    expand_chunk_action_blocks,
)
from .manifest import _slug, load_config_prompts, select_shard


def build_recurrent_capture_tasks(config_path: Path) -> list[dict[str, Any]]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompts = load_config_prompts(config_path, config)
    chunk_size = int(config["chunk_size"])
    denoising_steps = int(config["denoising_steps"])
    counterbalance_actions = config.get("counterbalance_actions")
    configured_schedules = config.get("command_block_schedules")
    schedule_assignment = str(config.get("schedule_assignment", "cross_product"))
    if schedule_assignment not in {"cross_product", "round_robin"}:
        raise ValueError("schedule_assignment must be cross_product or round_robin")
    if counterbalance_actions is not None:
        if configured_schedules is not None or "command_blocks" in config:
            raise ValueError("counterbalance_actions is exclusive with explicit schedules")
        actions = [str(value) for value in counterbalance_actions]
        event_count = int(config.get("counterbalance_event_count", 3))
        hold_blocks = int(config.get("counterbalance_hold_blocks", 1))
        hold_patterns = config.get("counterbalance_hold_patterns")
        cleanup_blocks = int(config.get("counterbalance_cleanup_blocks", 0))
        if len(set(actions)) < 2 or event_count <= 0 or hold_blocks <= 0 or cleanup_blocks < 0:
            raise ValueError("Invalid counterbalanced schedule configuration")
        paths = [
            list(path)
            for path in itertools.product(actions, repeat=event_count + 1)
            if all(left != right for left, right in zip(path, path[1:]))
        ]
        if hold_patterns is not None:
            parsed_patterns = [[int(value) for value in pattern] for pattern in hold_patterns]
            if not parsed_patterns or any(
                len(pattern) != event_count + 1 or any(value <= 0 for value in pattern)
                for pattern in parsed_patterns
            ):
                raise ValueError(
                    "Every counterbalance hold pattern must contain one positive "
                    "dwell per action segment"
                )
            command_schedules = [
                [
                    action
                    for action, dwell in zip(
                        path,
                        parsed_patterns[index % len(parsed_patterns)],
                        strict=True,
                    )
                    for _ in range(dwell)
                ]
                + [path[-1]] * cleanup_blocks
                for index, path in enumerate(paths)
            ]
        else:
            command_schedules = [
                [action for action in path for _ in range(hold_blocks)]
                + [path[-1]] * cleanup_blocks
                for path in paths
            ]
    elif configured_schedules is None:
        command_schedules = [[str(value) for value in config["command_blocks"]]]
    else:
        command_schedules = [
            [str(value) for value in schedule] for schedule in configured_schedules
        ]
        if not command_schedules:
            raise ValueError("command_block_schedules cannot be empty")
    receipt_steps = [int(value) for value in config["runtime_receipt_steps"]]
    rotate_receipts = bool(config.get("rotate_receipt_steps_by_seed", False))
    rotate_receipts_by_schedule = bool(config.get("rotate_receipt_steps_by_schedule", False))
    schedule_seed_stride = int(config.get("schedule_seed_stride", 0))
    prompt_seed_stride = int(config.get("prompt_seed_stride", 0))
    if schedule_seed_stride < 0 or prompt_seed_stride < 0:
        raise ValueError("seed strides must be nonnegative")
    jump_horizon = int(config["runtime_jump_horizon"])
    h1_events_per_refresh = int(config.get("runtime_h1_events_per_refresh", 0))
    if h1_events_per_refresh < 0:
        raise ValueError("runtime_h1_events_per_refresh must be nonnegative")
    for command_blocks in command_schedules:
        if len(command_blocks) < 2:
            raise ValueError("Recurrent capture requires at least two command blocks")
        invalid = sorted(set(command_blocks) - VALID_ACTIONS)
        if invalid:
            raise ValueError(f"Unknown recurrent actions: {invalid}")
    schedule_lengths = {len(schedule) for schedule in command_schedules}
    if schedule_assignment == "cross_product" and len(schedule_lengths) != 1:
        raise ValueError("Every command schedule must have the same length")
    if not receipt_steps or any(
        receipt <= 0 or receipt + jump_horizon >= denoising_steps for receipt in receipt_steps
    ):
        raise ValueError("Every recurrent receipt must leave one real DiT step")
    if "num_latent_frames" in config:
        configured_frames = int(config["num_latent_frames"])
        if any(configured_frames != chunk_size * length for length in schedule_lengths):
            raise ValueError("num_latent_frames must match command-block length")
    configured_event_blocks = config.get("event_block_indices")
    train_prompt_count = int(config.get("train_prompt_count", len(prompts)))
    if not 0 < train_prompt_count <= len(prompts):
        raise ValueError("train_prompt_count must select a nonempty prompt prefix")

    tasks: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_key = f"p{prompt_index:02d}-{_slug(prompt)}"
        indexed_schedules = list(enumerate(command_schedules))
        if schedule_assignment == "round_robin":
            indexed_schedules = [indexed_schedules[prompt_index % len(indexed_schedules)]]
        for schedule_index, command_blocks in indexed_schedules:
            num_frames = chunk_size * len(command_blocks)
            requested_commands = expand_chunk_action_blocks(command_blocks, chunk_size)
            all_stale_commands = build_event_stale_command_schedules(command_blocks, chunk_size)
            changed_blocks = {
                index
                for index in range(1, len(command_blocks))
                if command_blocks[index] != command_blocks[index - 1]
            }
            event_blocks = (
                sorted(changed_blocks)
                if configured_event_blocks is None
                else sorted({int(value) for value in configured_event_blocks})
            )
            if not event_blocks or any(
                index <= 0 or index >= len(command_blocks) for index in event_blocks
            ):
                raise ValueError("event_block_indices must select later video chunks")
            if set(event_blocks) != changed_blocks:
                raise ValueError("event_block_indices must exactly match command-block changes")
            event_pose_indices = [index * chunk_size for index in event_blocks]
            stale_commands_by_event = {
                pose: all_stale_commands[pose] for pose in event_pose_indices
            }
            for seed in config["seeds"]:
                effective_seed = (
                    int(seed)
                    + schedule_index * schedule_seed_stride
                    + prompt_index * prompt_seed_stride
                )
                task_receipt_steps = list(receipt_steps)
                if rotate_receipts:
                    offset = effective_seed % len(task_receipt_steps)
                    task_receipt_steps = task_receipt_steps[offset:] + task_receipt_steps[:offset]
                if rotate_receipts_by_schedule:
                    offset = schedule_index % len(task_receipt_steps)
                    task_receipt_steps = task_receipt_steps[offset:] + task_receipt_steps[:offset]
                key = (
                    f"{prompt_index}|{schedule_index}|{effective_seed}|"
                    f"{jump_horizon}|{command_blocks}"
                )
                digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
                sequence_id = (
                    f"{prompt_key}__q{schedule_index:02d}__s{effective_seed:03d}"
                    f"__recurrent__{digest}"
                )
                tasks.append(
                    {
                        "recurrent_sequence_id": sequence_id,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "master_id": f"cst-paper-p{prompt_index:03d}",
                        "dataset_split": (
                            "train" if prompt_index < train_prompt_count else "heldout"
                        ),
                        "command_schedule_index": schedule_index,
                        "schedule_assignment": schedule_assignment,
                        "seed": effective_seed,
                        "num_latent_frames": num_frames,
                        "chunk_size": chunk_size,
                        "denoising_steps": denoising_steps,
                        "command_blocks": command_blocks,
                        "stale_commands": stale_commands_by_event[event_pose_indices[0]],
                        "stale_commands_by_event": stale_commands_by_event,
                        "requested_commands": requested_commands,
                        "event_pose_indices": event_pose_indices,
                        "runtime_receipt_steps": task_receipt_steps,
                        "rotate_receipt_steps_by_seed": rotate_receipts,
                        "rotate_receipt_steps_by_schedule": (rotate_receipts_by_schedule),
                        "schedule_seed_stride": schedule_seed_stride,
                        "prompt_seed_stride": prompt_seed_stride,
                        "runtime_jump_horizon": jump_horizon,
                        "runtime_h1_events_per_refresh": h1_events_per_refresh,
                        "save_cleanup_artifact": bool(config.get("save_cleanup_artifact", False)),
                        "save_trajectory_artifact": bool(
                            config.get("save_trajectory_artifact", False)
                        ),
                    }
                )
    return tasks


__all__ = ["build_recurrent_capture_tasks", "select_shard"]
