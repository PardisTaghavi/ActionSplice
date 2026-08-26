"""Build deterministic Phase-1 experiment tasks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..backends.conditioning import (
    INJECTION_AFTER_DENOISING_STEPS,
    SCENARIOS,
    build_command_pair,
    validate_experiment_geometry,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_experiment_geometry(
        num_latent_frames=int(config["num_latent_frames"]),
        chunk_size=int(config["chunk_size"]),
        event_pose_index=int(config["event_pose_index"]),
    )
    if int(config["denoising_steps"]) != 4:
        raise ValueError("Phase 1 currently targets minWM's four-step Action2V checkpoint")
    return config


def load_prompts(config_path: Path, prompt_file: str) -> list[str]:
    path = Path(prompt_file)
    if not path.is_absolute():
        path = config_path.parent.parent / path
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_config_prompts(
    config_path: Path,
    config: dict[str, Any],
) -> list[str]:
    """Load one prompt file or an ordered collection of prompt files."""
    if "prompt_files" in config:
        if "prompt_file" in config:
            raise ValueError("Specify prompt_file or prompt_files, not both")
        files = [str(value) for value in config["prompt_files"]]
        if not files:
            raise ValueError("prompt_files cannot be empty")
        prompts = [
            prompt for prompt_file in files for prompt in load_prompts(config_path, prompt_file)
        ]
    else:
        prompts = load_prompts(config_path, str(config["prompt_file"]))
    if len(set(prompts)) != len(prompts):
        raise ValueError("Prompt files must not contain duplicate prompts")
    return prompts


def _slug(text: str, max_length: int = 42) -> str:
    compact = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return compact[:max_length].rstrip("-") or "prompt"


def build_tasks(config_path: Path) -> list[dict[str, Any]]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    prompts = load_prompts(config_path, str(config["prompt_file"]))
    chunk_size = int(config["chunk_size"])
    event_pose_index = int(config["event_pose_index"])
    num_latent_frames = int(config["num_latent_frames"])

    tasks: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_key = f"p{prompt_index:02d}-{_slug(prompt)}"
        for seed in config["seeds"]:
            for scenario_name in config["scenarios"]:
                scenario = SCENARIOS[scenario_name]
                injections = ["control"] if scenario.is_control else config["injections"]
                for injection in injections:
                    if injection not in INJECTION_AFTER_DENOISING_STEPS:
                        raise ValueError(f"Unknown injection {injection!r}")

                    # If the action is delivered only after the event chunk,
                    # its desired trajectory begins at the following chunk.
                    effective_pose_index = (
                        event_pose_index + chunk_size
                        if injection == "after_chunk"
                        else event_pose_index
                    )
                    _, requested_commands = build_command_pair(
                        scenario_name=scenario_name,
                        num_latent_frames=num_latent_frames,
                        effective_pose_index=event_pose_index,
                    )
                    stale_commands, desired_commands = build_command_pair(
                        scenario_name=scenario_name,
                        num_latent_frames=num_latent_frames,
                        effective_pose_index=effective_pose_index,
                    )
                    stable_key = "|".join(
                        [
                            str(prompt_index),
                            str(seed),
                            scenario_name,
                            injection,
                            str(event_pose_index),
                        ]
                    )
                    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:10]
                    task_id = (
                        f"{prompt_key}__s{int(seed):03d}__{scenario_name}__{injection}__{digest}"
                    )
                    tasks.append(
                        {
                            "task_id": task_id,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "seed": int(seed),
                            "scenario": scenario_name,
                            "scenario_description": scenario.description,
                            "analysis_scope": ("stress" if scenario.is_stress else "primary"),
                            "old_action": scenario.old_action,
                            "new_action": scenario.new_action,
                            "injection": injection,
                            "switch_after_denoising_steps": INJECTION_AFTER_DENOISING_STEPS[
                                injection
                            ],
                            "action_receipt_after_denoising_steps": (
                                None
                                if injection == "control"
                                else 4
                                if injection == "after_chunk"
                                else INJECTION_AFTER_DENOISING_STEPS[injection]
                            ),
                            "event_pose_index": event_pose_index,
                            "event_chunk_index": event_pose_index // chunk_size,
                            "effective_pose_index": effective_pose_index,
                            "num_latent_frames": num_latent_frames,
                            "chunk_size": chunk_size,
                            "denoising_steps": int(config["denoising_steps"]),
                            "fps": int(config["fps"]),
                            "stale_commands": stale_commands,
                            # Requested is the ideal response beginning when the
                            # user action arrives. Desired is the trajectory
                            # actually supplied when the selected strategy
                            # applies that action.
                            "requested_commands": requested_commands,
                            "desired_commands": desired_commands,
                        }
                    )
    return tasks


def select_shard(
    tasks: list[dict[str, Any]],
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    # Keep all conditions for a prompt together so one model load produces a
    # complete visual comparison set.
    return [task for task in tasks if int(task["prompt_index"]) % num_shards == shard_index]
