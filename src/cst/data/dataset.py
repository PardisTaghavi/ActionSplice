"""Datasets backed by matched counterfactual transport captures."""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CaptureSplit:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]


def discover_capture_paths(capture_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(capture_dir.resolve().glob("*.pt")))
    if not paths:
        raise FileNotFoundError(f"No transport captures found in {capture_dir}")
    return paths


def split_capture_paths(
    paths: Iterable[Path],
    *,
    validation_fraction: float,
    seed: int,
) -> CaptureSplit:
    """Split by capture, never by receipt pair from the same capture."""
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("At least two captures are required for a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    random.Random(seed).shuffle(paths)
    validation_count = max(1, round(len(paths) * validation_fraction))
    validation_count = min(validation_count, len(paths) - 1)
    return CaptureSplit(
        train=tuple(paths[validation_count:]),
        validation=tuple(paths[:validation_count]),
    )


def split_grouped_capture_paths(
    paths: Iterable[Path],
    *,
    validation_fraction: float,
    seed: int,
    group_by: str,
) -> CaptureSplit:
    """Two-way split that keeps every scene/prompt group intact."""
    from .partitions import capture_group_id

    paths = tuple(sorted(Path(path).resolve() for path in paths))
    if len(paths) < 2:
        raise ValueError("At least two captures are required for a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        group = capture_group_id(path, group_by=group_by)
        grouped.setdefault(group, []).append(path)
    group_ids = sorted(grouped)
    if len(group_ids) < 2:
        raise ValueError("Grouped split requires at least two groups")
    random.Random(seed).shuffle(group_ids)
    validation_count = max(
        1,
        round(len(group_ids) * validation_fraction),
    )
    validation_count = min(validation_count, len(group_ids) - 1)
    validation_groups = set(group_ids[:validation_count])
    validation = tuple(
        path for group in group_ids if group in validation_groups for path in grouped[group]
    )
    train = tuple(
        path for group in group_ids if group not in validation_groups for path in grouped[group]
    )
    return CaptureSplit(train=train, validation=validation)


def split_capture_paths_by_metadata(
    paths: Iterable[Path],
    *,
    field: str,
    train_value: str = "train",
    validation_value: str = "heldout",
    group_by: str | None = "prompt",
) -> CaptureSplit:
    """Use an explicit capture-sidecar split and reject group leakage."""
    from .partitions import capture_group_id, load_capture_metadata

    paths = tuple(sorted(Path(path).resolve() for path in paths))
    selected: dict[str, list[Path]] = {
        str(train_value): [],
        str(validation_value): [],
    }
    if train_value == validation_value:
        raise ValueError("Train and validation metadata values must differ")
    for path in paths:
        value = str(load_capture_metadata(path).get(field, "")).strip()
        if value not in selected:
            raise ValueError(f"Capture {path.name} has unsupported {field}={value!r}")
        selected[value].append(path)
    if not selected[train_value] or not selected[validation_value]:
        raise ValueError("Explicit metadata split produced an empty partition")
    split = CaptureSplit(
        train=tuple(selected[train_value]),
        validation=tuple(selected[validation_value]),
    )
    if group_by:
        train_groups = {capture_group_id(path, group_by=group_by) for path in split.train}
        validation_groups = {capture_group_id(path, group_by=group_by) for path in split.validation}
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(
                f"Explicit metadata split leaks {group_by} groups: {sorted(overlap)[:3]}"
            )
    return split


def estimate_reference_nfe_ms(paths: Iterable[Path]) -> float:
    """Estimate one generator-call latency from saved event-branch traces."""
    paths = tuple(Path(path).resolve() for path in paths)
    elapsed: list[float] = []
    # Recurrent capture stores one timing trace per generated sequence rather
    # than duplicating it into every event capture.
    sequence_dirs = {path.parent.parent / "sequences" for path in paths}
    for sequence_dir in sequence_dirs:
        for summary_path in sorted(sequence_dir.glob("*.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for event in summary.get("generator_events", []):
                phase = str(event.get("phase", ""))
                branch = str(event.get("branch", ""))
                if not (
                    phase.startswith(("old_teacher_", "rollback_teacher_"))
                    or branch in {"old_teacher", "rollback_teacher"}
                ):
                    continue
                step_index = event.get("step_index", event.get("denoise_step_index"))
                if step_index is None:
                    continue
                value = float(event["elapsed_ms"])
                if value > 0.0:
                    elapsed.append(value)
    if elapsed:
        return statistics.median(elapsed)

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for event in payload.get("metadata", {}).get("events", []):
            if (
                event.get("branch") in {"old", "new"}
                and event.get("denoise_step_index") is not None
            ):
                value = float(event["elapsed_ms"])
                if value > 0.0:
                    elapsed.append(value)
    if not elapsed:
        raise ValueError("Captures contain no event-branch NFE timings")
    return statistics.median(elapsed)


class TransportPairDataset(Dataset[dict[str, Any]]):
    """Expose every legal receipt state from each trajectory capture."""

    def __init__(
        self,
        capture_paths: Iterable[Path],
        *,
        jump_horizons: Iterable[int] = (0,),
        receipt_steps: Iterable[int] | None = None,
        source_branch: str = "old",
        target_branch: str = "new",
        recurrent_max_event_index: int | None = None,
        recurrent_max_rollout_age: int | None = None,
    ) -> None:
        self.capture_paths = tuple(Path(path).resolve() for path in capture_paths)
        if not self.capture_paths:
            raise ValueError("TransportPairDataset requires at least one capture")
        self.jump_horizons = tuple(sorted({int(horizon) for horizon in jump_horizons}))
        if self.jump_horizons != (0,):
            raise ValueError("ActionSplice datasets require same-step targets")
        self.receipt_steps = (
            None if receipt_steps is None else tuple(sorted({int(step) for step in receipt_steps}))
        )
        if self.receipt_steps is not None and (
            not self.receipt_steps or self.receipt_steps[0] <= 0
        ):
            raise ValueError("receipt_steps must contain positive integers")
        if source_branch not in {"old", "new"}:
            raise ValueError("source_branch must be old or new")
        if target_branch not in {"old", "new"}:
            raise ValueError("target_branch must be old or new")
        self.source_branch = source_branch
        self.target_branch = target_branch
        self.recurrent_max_event_index = (
            None if recurrent_max_event_index is None else int(recurrent_max_event_index)
        )
        if self.recurrent_max_event_index is not None and self.recurrent_max_event_index < 0:
            raise ValueError("recurrent_max_event_index must be nonnegative")
        self.recurrent_max_rollout_age = (
            None if recurrent_max_rollout_age is None else int(recurrent_max_rollout_age)
        )
        if self.recurrent_max_rollout_age is not None and self.recurrent_max_rollout_age < 0:
            raise ValueError("recurrent_max_rollout_age must be nonnegative")
        self.index: list[tuple[int, int, int]] = []
        self._payload_cache: dict[int, dict[str, Any]] = {}
        for path_index, path in enumerate(self.capture_paths):
            payload = _load_payload(path)
            metadata = payload["metadata"]
            if self.recurrent_max_event_index is not None:
                event_index = metadata.get("recurrent_event_index")
                if event_index is None:
                    raise ValueError("Recurrent curriculum requires recurrent_event_index")
                if int(event_index) > self.recurrent_max_event_index:
                    continue
            if self.recurrent_max_rollout_age is not None:
                rollout_age = metadata.get("recurrent_rollout_age")
                if rollout_age is None:
                    raise ValueError("Rollout-age curriculum requires recurrent_rollout_age")
                if int(rollout_age) > self.recurrent_max_rollout_age:
                    continue
            denoising_steps = int(metadata["denoising_steps"])
            for receipt_step in metadata["receipt_steps"]:
                receipt_step = int(receipt_step)
                if self.receipt_steps is not None and receipt_step not in self.receipt_steps:
                    continue
                for jump_horizon in self.jump_horizons:
                    if receipt_step + jump_horizon <= denoising_steps:
                        self.index.append((path_index, receipt_step, jump_horizon))
        if not self.index:
            raise ValueError("Captures contain no legal receipt steps")

    def __len__(self) -> int:
        return len(self.index)

    def balance_key(self, index: int, fields: Iterable[str]) -> tuple[Any, ...]:
        """Return metadata factors used for inverse-frequency sampling."""
        path_index, receipt_step, jump_horizon = self.index[index]
        metadata = self._payload(path_index)["metadata"]
        values: list[Any] = []
        for field in fields:
            if field == "receipt_step":
                value = receipt_step
            elif field == "jump_horizon":
                value = jump_horizon
            elif field == "action_transition":
                value = _optional_action_transition(metadata)
                if value == "unknown":
                    raise ValueError(
                        "action_transition balancing requires old/new actions "
                        "or command-block event metadata"
                    )
            else:
                if field not in metadata:
                    raise ValueError(f"Training balance field {field!r} is absent from capture")
                value = metadata[field]
            values.append(value)
        return tuple(values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path_index, receipt_step, jump_horizon = self.index[index]
        payload = self._payload(path_index)
        tensors = payload["tensors"]
        metadata = payload["metadata"]
        source_states = tensors[f"{self.source_branch}_states"]
        target_states = tensors[f"{self.target_branch}_states"]
        if int(source_states.shape[1]) != 1:
            raise ValueError("Initial implementation requires capture batch size 1")
        denoising_steps = int(source_states.shape[0])
        if not 0 < receipt_step < denoising_steps:
            raise ValueError("Receipt step is outside the saved trajectory")
        target_step = receipt_step + jump_horizon
        target_predictions = tensors.get(f"{self.target_branch}_predictions")
        source_predictions = tensors.get(f"{self.source_branch}_predictions")
        transition_noises = tensors["transition_noises"]
        if target_predictions is None or source_predictions is None:
            raise ValueError("Transport targets require saved predictions")
        deterministic_state_targets = metadata.get(
            "transport_target_parameterization"
        ) == "state" and not bool(metadata.get("stochastic_transitions", True))
        receipt_target_state = target_states[receipt_step, 0]
        if deterministic_state_targets:
            receipt_clean_target = receipt_target_state
            receipt_transition_noise = torch.zeros_like(receipt_target_state)
            receipt_sigma = torch.tensor(0.0, dtype=torch.float32)
        else:
            receipt_clean_target = target_predictions[receipt_step - 1, 0]
            receipt_transition_noise = transition_noises[receipt_step - 1, 0]
            receipt_sigma = _infer_transition_sigma(
                receipt_clean_target,
                receipt_transition_noise,
                receipt_target_state,
            )
        if target_step < denoising_steps:
            target_state = target_states[target_step, 0]
            if deterministic_state_targets:
                clean_target = target_state
                target_transition_noise = torch.zeros_like(target_state)
                target_sigma = torch.tensor(0.0, dtype=torch.float32)
            else:
                clean_target = target_predictions[target_step - 1, 0]
                target_transition_noise = transition_noises[target_step - 1, 0]
                target_sigma = _infer_transition_sigma(
                    clean_target,
                    target_transition_noise,
                    target_state,
                )
        elif target_step == denoising_steps:
            target_state = target_predictions[-1, 0]
            clean_target = target_state
            target_transition_noise = torch.zeros_like(target_state)
            target_sigma = torch.tensor(0.0, dtype=torch.float32)
        else:
            raise ValueError("Jump target exceeds the denoising schedule")

        rollout_clean_targets = []
        rollout_target_states = []
        rollout_transition_noises = []
        rollout_sigmas = []
        for rollout_step in range(receipt_step + 1, target_step + 1):
            if rollout_step < denoising_steps:
                rollout_state = target_states[rollout_step, 0]
                if deterministic_state_targets:
                    rollout_clean = rollout_state
                    rollout_noise = torch.zeros_like(rollout_state)
                    rollout_sigma = torch.tensor(0.0, dtype=torch.float32)
                else:
                    rollout_clean = target_predictions[rollout_step - 1, 0]
                    rollout_noise = transition_noises[rollout_step - 1, 0]
                    rollout_sigma = _infer_transition_sigma(
                        rollout_clean,
                        rollout_noise,
                        rollout_state,
                    )
            else:
                rollout_state = target_predictions[-1, 0]
                rollout_clean = rollout_state
                rollout_noise = torch.zeros_like(rollout_state)
                rollout_sigma = torch.tensor(0.0, dtype=torch.float32)
            rollout_clean_targets.append(rollout_clean.float())
            rollout_target_states.append(rollout_state.float())
            rollout_transition_noises.append(rollout_noise.float())
            rollout_sigmas.append(rollout_sigma.float())

        history_tail = tensors["history_tail"]
        active_shape = source_states[receipt_step, 0].shape
        if int(history_tail.shape[1]) == 0:
            history = torch.zeros_like(source_states[receipt_step, 0])
        else:
            history = history_tail[0]
        if history.shape != active_shape:
            raise ValueError("History tail and active state must share the same latent shape")

        sample = {
            "active_state": source_states[receipt_step, 0].float(),
            "cached_prediction": tensors[f"{self.source_branch}_predictions"][
                receipt_step - 1, 0
            ].float(),
            "initial_state": source_states[0, 0].float(),
            "history_tail": history.float(),
            "noise_trace": tensors["transition_noises"][:, 0].float(),
            "old_viewmats": tensors[f"{self.source_branch}_viewmats"][0].float(),
            "new_viewmats": tensors[f"{self.target_branch}_viewmats"][0].float(),
            "intrinsics": tensors["intrinsics"][0].float(),
            "receipt_step": torch.tensor(receipt_step, dtype=torch.long),
            "jump_horizon": torch.tensor(jump_horizon, dtype=torch.long),
            "target_step": torch.tensor(target_step, dtype=torch.long),
            "target_state": target_state.float(),
            "clean_target": clean_target.float(),
            "target_transition_noise": target_transition_noise.float(),
            "target_sigma": target_sigma.float(),
            "rollout_clean_targets": (
                torch.stack(rollout_clean_targets)
                if rollout_clean_targets
                else target_state.new_empty((0, *target_state.shape)).float()
            ),
            "rollout_target_states": (
                torch.stack(rollout_target_states)
                if rollout_target_states
                else target_state.new_empty((0, *target_state.shape)).float()
            ),
            "rollout_transition_noises": (
                torch.stack(rollout_transition_noises)
                if rollout_transition_noises
                else target_state.new_empty((0, *target_state.shape)).float()
            ),
            "rollout_sigmas": (
                torch.stack(rollout_sigmas)
                if rollout_sigmas
                else torch.empty(0, dtype=torch.float32)
            ),
            "receipt_target_state": receipt_target_state.float(),
            "receipt_clean_target": receipt_clean_target.float(),
            "receipt_transition_noise": receipt_transition_noise.float(),
            "receipt_sigma": receipt_sigma.float(),
            "capture_index": torch.tensor(path_index, dtype=torch.long),
            "recurrent_event_index": torch.tensor(
                int(payload["metadata"].get("recurrent_event_index", 0)),
                dtype=torch.long,
            ),
            "rollout_age": torch.tensor(
                int(payload["metadata"].get("recurrent_rollout_age", 0)),
                dtype=torch.long,
            ),
            "action_transition": _optional_action_transition(metadata),
        }
        intra_chunk_offset = metadata.get("intra_chunk_offset")
        if intra_chunk_offset is not None:
            offset = int(intra_chunk_offset)
            frame_count = int(active_shape[0])
            if not 0 < offset < frame_count:
                raise ValueError("intra_chunk_offset must select a nonempty suffix")
            suffix_mask = torch.zeros(
                frame_count,
                1,
                1,
                1,
                dtype=torch.float32,
            )
            suffix_mask[offset:] = 1.0
            sample["temporal_suffix_mask"] = suffix_mask
            sample["intra_chunk_offset"] = torch.tensor(offset, dtype=torch.long)
        return sample

    def _payload(self, path_index: int) -> dict[str, Any]:
        cached = self._payload_cache.get(path_index)
        if cached is not None:
            return cached
        payload = _load_payload(self.capture_paths[path_index])
        # One payload per worker is enough to reuse adjacent receipt examples
        # without allowing memory to grow with the full dataset.
        self._payload_cache.clear()
        self._payload_cache[path_index] = payload
        return payload


def _optional_action_transition(metadata: dict[str, Any]) -> str:
    """Describe an action change when recurrent metadata is available."""
    old_action = metadata.get("old_action")
    new_action = metadata.get("new_action")
    if old_action is not None and new_action is not None:
        return f"{old_action}->{new_action}"
    blocks = metadata.get("command_blocks")
    event_pose = metadata.get("event_pose_index")
    chunk_size = metadata.get("chunk_size")
    if blocks is not None and event_pose is not None and chunk_size is not None:
        block_index = int(event_pose) // int(chunk_size)
        if 0 < block_index < len(blocks):
            return f"{blocks[block_index - 1]}->{blocks[block_index]}"
    return "unknown"


class RecurrentSequenceBatchSampler:
    """Batch contiguous events from one recurrent teacher trajectory."""

    def __init__(
        self,
        dataset: TransportPairDataset,
        *,
        sequence_length: int,
        shuffle: bool,
        seed: int,
        exclude_scheduled_refresh: bool = True,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = int(sequence_length)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[tuple[str, int, int, int], list[tuple[int, int]]] = {}
        for sample_index, (path_index, receipt_step, horizon) in enumerate(dataset.index):
            payload = dataset._payload(path_index)
            metadata = payload["metadata"]
            sequence_id = metadata.get("recurrent_sequence_id")
            event_index = metadata.get("recurrent_event_index")
            if sequence_id is None or event_index is None:
                raise ValueError("Sequence-window training requires recurrent capture metadata")
            if exclude_scheduled_refresh and bool(metadata.get("runtime_scheduled_refresh", False)):
                continue
            refresh_cycle = int(metadata.get("recurrent_refresh_cycle", 0))
            key = (
                str(sequence_id),
                int(receipt_step),
                int(horizon),
                refresh_cycle,
            )
            grouped.setdefault(key, []).append((int(event_index), sample_index))

        batches: list[tuple[int, ...]] = []
        for events in grouped.values():
            ordered = sorted(events)
            for start in range(0, len(ordered), self.sequence_length):
                window = ordered[start : start + self.sequence_length]
                if len(window) != self.sequence_length:
                    continue
                event_indices = [event_index for event_index, _ in window]
                if any(
                    right != left + 1
                    for left, right in zip(
                        event_indices,
                        event_indices[1:],
                    )
                ):
                    continue
                batches.append(tuple(sample_index for _, sample_index in window))
        if not batches:
            raise ValueError(
                "Recurrent captures contain no complete sequence windows of "
                f"length {self.sequence_length}"
            )
        self.batches = tuple(batches)

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def _infer_transition_sigma(
    clean_prediction: torch.Tensor,
    transition_noise: torch.Tensor,
    noisy_state: torch.Tensor,
) -> torch.Tensor:
    """Infer scalar sigma in z=(1-sigma)*x0+sigma*epsilon."""
    clean = clean_prediction.float()
    noise_direction = transition_noise.float() - clean
    state_direction = noisy_state.float() - clean
    denominator = noise_direction.square().sum().clamp_min(1e-12)
    sigma = (state_direction * noise_direction).sum() / denominator
    return sigma.detach()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported transport capture schema in {path}")
    if "metadata" not in payload or "tensors" not in payload:
        raise ValueError(f"Malformed transport capture {path}")
    return payload
