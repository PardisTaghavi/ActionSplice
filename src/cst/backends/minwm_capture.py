"""Matched active-state capture for counterfactual transport training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .minwm_cache import (
    _cache_index_snapshot,
    _reset_pipeline_caches,
    validate_cache_overwrite_trace,
)

SCHEMA_VERSION = 1


@dataclass
class TransportCapture:
    """One matched old/new event-chunk trajectory pair."""

    tensors: dict[str, Any]
    metadata: dict[str, Any]


def capture_transport_pair(
    *,
    pipeline: Any,
    noise: Any,
    text_prompt: str,
    stale_viewmats: Any,
    requested_viewmats: Any,
    intrinsics: Any,
    event_pose_index: int,
    intra_chunk_offset: int = 0,
    trace_cache_indices: bool = False,
) -> TransportCapture:
    """Capture matched old/new sampler states at every legal receipt step.

    Prior chunks are generated once under the shared stale condition. The
    event chunk is then evaluated twice from the same initial active noise,
    committed cache, and transition-noise bank. Repeated calls overwrite the
    active cache slots; committed history is never changed.
    """
    import torch

    if pipeline.independent_first_frame:
        raise ValueError(
            "Transport capture currently requires independent_first_frame=False"
        )
    batch_size, num_frames, _, _, _ = noise.shape
    chunk_size = int(pipeline.num_frame_per_block)
    denoising_steps = len(pipeline.denoising_step_list)
    if denoising_steps < 2:
        raise ValueError("Transport capture requires at least two denoising steps")
    if num_frames % chunk_size:
        raise ValueError("Noise frame count must be divisible by chunk size")
    if event_pose_index % chunk_size:
        raise ValueError("event_pose_index must be a chunk boundary")
    if intra_chunk_offset and not 0 < intra_chunk_offset < chunk_size:
        raise ValueError("intra_chunk_offset must select a nonempty chunk suffix")
    if not 0 <= event_pose_index < num_frames:
        raise ValueError("event_pose_index must identify an existing chunk")
    for name, tensor in (
        ("stale_viewmats", stale_viewmats),
        ("requested_viewmats", requested_viewmats),
        ("intrinsics", intrinsics),
    ):
        if int(tensor.shape[0]) != batch_size or int(tensor.shape[1]) < num_frames:
            raise ValueError(
                f"{name} must cover batch={batch_size}, frames={num_frames}"
            )

    conditional_dict = pipeline.text_encoder(text_prompts=[text_prompt])
    _reset_pipeline_caches(pipeline, noise)
    events: list[dict[str, Any]] = []
    output = torch.zeros_like(noise)
    history_tail = output[:, :0]
    event_transition_noises: list[Any] | None = None

    def synchronize(tensor: Any) -> None:
        if bool(getattr(tensor, "is_cuda", False)):
            torch.cuda.synchronize(tensor.device)

    def camera_slice(tensor: Any, start: int, count: int) -> Any:
        return tensor[:, start : start + count]

    def model_timestep(value: Any, count: int) -> Any:
        return (
            torch.ones(
                [batch_size, count],
                device=noise.device,
                dtype=torch.int64,
            )
            * value
        )

    def call_generator(
        *,
        active: Any,
        timestep: Any,
        start_frame: int,
        viewmats: Any,
        branch: str,
        step_index: int | None,
    ) -> Any:
        before = _cache_index_snapshot(pipeline) if trace_cache_indices else None
        synchronize(active)
        started = time.perf_counter()
        result = pipeline.generator(
            noisy_image_or_video=active,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=start_frame * pipeline.frame_seq_length,
            viewmats=camera_slice(viewmats, start_frame, active.shape[1]),
            Ks=camera_slice(intrinsics, start_frame, active.shape[1]),
            prope_kv_cache=pipeline.prope_kv_cache1,
        )
        synchronize(active)
        ended = time.perf_counter()
        after = _cache_index_snapshot(pipeline) if trace_cache_indices else None
        events.append(
            {
                "branch": branch,
                "start_frame": start_frame,
                "frame_count": int(active.shape[1]),
                "denoise_step_index": step_index,
                "phase": (
                    "context_update"
                    if step_index is None
                    else f"{branch}_denoise_{step_index + 1}"
                ),
                "elapsed_ms": (ended - started) * 1000.0,
                "cache_indices_before": before,
                "cache_indices_after": after,
            }
        )
        return result

    def transition_noise_bank(active: Any) -> list[Any]:
        return [
            torch.randn_like(active.flatten(0, 1)) for _ in range(denoising_steps - 1)
        ]

    def add_transition_noise(
        prediction: Any,
        transition_noise: Any,
        next_step_index: int,
    ) -> Any:
        next_timestep = pipeline.denoising_step_list[next_step_index]
        frame_count = int(prediction.shape[1])
        return pipeline.scheduler.add_noise(
            prediction.flatten(0, 1),
            transition_noise,
            next_timestep
            * torch.ones(
                [batch_size * frame_count],
                device=noise.device,
                dtype=torch.long,
            ),
        ).unflatten(0, prediction.shape[:2])

    def run_active_branch(
        *,
        initial_active: Any,
        transition_noises: list[Any],
        branch_viewmats: Any,
        branch: str,
        prefix_reference_states: list[Any] | None = None,
        prefix_reference_predictions: list[Any] | None = None,
        prefix_length: int = 0,
    ) -> tuple[list[Any], list[Any]]:
        states = [initial_active.detach().clone()]
        predictions: list[Any] = []
        active = initial_active
        frame_count = int(initial_active.shape[1])
        for step_index, timestep_value in enumerate(pipeline.denoising_step_list):
            _, prediction = call_generator(
                active=active,
                timestep=model_timestep(timestep_value, frame_count),
                start_frame=event_pose_index,
                viewmats=branch_viewmats,
                branch=branch,
                step_index=step_index,
            )
            if prefix_length:
                if prefix_reference_predictions is None:
                    raise ValueError("Prefix clamping requires reference predictions")
                prediction = torch.cat(
                    [
                        prefix_reference_predictions[step_index][:, :prefix_length],
                        prediction[:, prefix_length:],
                    ],
                    dim=1,
                )
            predictions.append(prediction.detach().clone())
            if step_index < denoising_steps - 1:
                active = add_transition_noise(
                    prediction,
                    transition_noises[step_index],
                    step_index + 1,
                )
                if prefix_length:
                    if prefix_reference_states is None:
                        raise ValueError("Prefix clamping requires reference states")
                    active = torch.cat(
                        [
                            prefix_reference_states[step_index + 1][:, :prefix_length],
                            active[:, prefix_length:],
                        ],
                        dim=1,
                    )
                states.append(active.detach().clone())
        return states, predictions

    with torch.no_grad():
        for start_frame in range(0, event_pose_index, chunk_size):
            frame_count = min(chunk_size, num_frames - start_frame)
            active = noise[:, start_frame : start_frame + frame_count]
            prior_noises = transition_noise_bank(active)
            prediction = None
            for step_index, timestep_value in enumerate(pipeline.denoising_step_list):
                _, prediction = call_generator(
                    active=active,
                    timestep=model_timestep(timestep_value, frame_count),
                    start_frame=start_frame,
                    viewmats=stale_viewmats,
                    branch="shared_prefix",
                    step_index=step_index,
                )
                if step_index < denoising_steps - 1:
                    active = add_transition_noise(
                        prediction,
                        prior_noises[step_index],
                        step_index + 1,
                    )
            if prediction is None:
                raise RuntimeError("Shared prefix produced no prediction")
            output[:, start_frame : start_frame + frame_count] = prediction
            history_tail = prediction.detach().clone()
            call_generator(
                active=prediction,
                timestep=model_timestep(
                    pipeline.args.context_noise,
                    frame_count,
                ),
                start_frame=start_frame,
                viewmats=stale_viewmats,
                branch="shared_prefix",
                step_index=None,
            )

        initial_active = noise[:, event_pose_index : event_pose_index + chunk_size]
        event_transition_noises = transition_noise_bank(initial_active)
        old_states, old_predictions = run_active_branch(
            initial_active=initial_active,
            transition_noises=event_transition_noises,
            branch_viewmats=stale_viewmats,
            branch="old",
        )
        new_states, new_predictions = run_active_branch(
            initial_active=initial_active,
            transition_noises=event_transition_noises,
            branch_viewmats=requested_viewmats,
            branch="new",
            prefix_reference_states=old_states,
            prefix_reference_predictions=old_predictions,
            prefix_length=intra_chunk_offset,
        )

    if event_transition_noises is None:
        raise RuntimeError("Event transition-noise bank was not created")
    old_state_tensor = torch.stack(old_states, dim=0)
    new_state_tensor = torch.stack(new_states, dim=0)
    if not torch.equal(old_state_tensor[0], new_state_tensor[0]):
        raise AssertionError("Old/new branches did not share the initial state")

    tensors = {
        "old_states": old_state_tensor,
        "new_states": new_state_tensor,
        "old_predictions": torch.stack(old_predictions, dim=0),
        "new_predictions": torch.stack(new_predictions, dim=0),
        "transition_noises": torch.stack(
            [
                value.unflatten(0, initial_active.shape[:2])
                for value in event_transition_noises
            ],
            dim=0,
        ),
        "history_tail": history_tail,
        "old_viewmats": camera_slice(
            stale_viewmats,
            event_pose_index,
            chunk_size,
        )
        .detach()
        .clone(),
        "new_viewmats": camera_slice(
            requested_viewmats,
            event_pose_index,
            chunk_size,
        )
        .detach()
        .clone(),
        "intrinsics": camera_slice(
            intrinsics,
            event_pose_index,
            chunk_size,
        )
        .detach()
        .clone(),
        "timesteps": pipeline.denoising_step_list.detach().clone(),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "text_prompt": text_prompt,
        "event_pose_index": event_pose_index,
        "effective_pose_index": event_pose_index + intra_chunk_offset,
        "intra_chunk_offset": intra_chunk_offset,
        "teacher_prefix_clamped": bool(intra_chunk_offset),
        "chunk_size": chunk_size,
        "denoising_steps": denoising_steps,
        "receipt_steps": list(range(1, denoising_steps)),
        "shared_prefix_chunk_count": event_pose_index // chunk_size,
        "shared_prefix_generator_calls": sum(
            event["branch"] == "shared_prefix" for event in events
        ),
        "event_generator_calls": sum(
            event["branch"] in {"old", "new"} for event in events
        ),
        "generator_elapsed_ms": sum(float(event["elapsed_ms"]) for event in events),
        "events": events,
        "cache_overwrite_validation": (
            validate_cache_overwrite_trace(
                events,
                frame_seq_length=int(pipeline.frame_seq_length),
            )
            if trace_cache_indices
            else None
        ),
    }
    validate_transport_capture(TransportCapture(tensors=tensors, metadata=metadata))
    return TransportCapture(tensors=tensors, metadata=metadata)


def validate_transport_capture(capture: TransportCapture) -> None:
    """Validate schema and tensor alignment without evaluating semantics."""
    import torch

    tensors = capture.tensors
    required = {
        "old_states",
        "new_states",
        "old_predictions",
        "new_predictions",
        "transition_noises",
        "history_tail",
        "old_viewmats",
        "new_viewmats",
        "intrinsics",
        "timesteps",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(f"Transport capture is missing tensors: {missing}")
    denoising_steps = int(capture.metadata["denoising_steps"])
    old_states = tensors["old_states"]
    new_states = tensors["new_states"]
    if old_states.shape != new_states.shape:
        raise ValueError("Old/new state trajectories have different shapes")
    if int(old_states.shape[0]) != denoising_steps:
        raise ValueError("State trajectories must contain z_0,...,z_(K-1)")
    for name in ("old_predictions", "new_predictions"):
        if int(tensors[name].shape[0]) != denoising_steps:
            raise ValueError(f"{name} must contain one prediction per step")
    if int(tensors["transition_noises"].shape[0]) != denoising_steps - 1:
        raise ValueError("Transition-noise trace has the wrong length")
    if int(tensors["timesteps"].numel()) != denoising_steps:
        raise ValueError("Timestep trace has the wrong length")
    if not torch.equal(old_states[0], new_states[0]):
        raise ValueError("Old/new trajectories must start from identical z_0")
    expected_receipts = list(range(1, denoising_steps))
    if list(capture.metadata["receipt_steps"]) != expected_receipts:
        raise ValueError("Receipt-step metadata is inconsistent")


def save_transport_capture(path: Path, capture: TransportCapture) -> None:
    """Persist a capture with all tensors detached on CPU."""
    import torch

    validate_transport_capture(capture)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": capture.metadata,
        "tensors": {
            name: tensor.detach().cpu() for name, tensor in capture.tensors.items()
        },
    }
    torch.save(payload, path)
