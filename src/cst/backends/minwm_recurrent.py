"""Recurrent on-policy CST-R capture for an unmodified minWM pipeline."""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..core.runtime import apply_transport_model
from ..data.capture import TransportCapture, validate_transport_capture
from .minwm_cache import _reset_pipeline_caches


def capture_recurrent_cst_r(
    *,
    pipeline: Any,
    noise: Any,
    text_prompt: str,
    stale_viewmats_by_event: dict[int, Any],
    requested_viewmats: Any,
    intrinsics: Any,
    event_pose_indices: Sequence[int],
    receipt_steps: Sequence[int],
    corrector: Any,
) -> tuple[list[TransportCapture], Any, dict[str, Any]]:
    """Capture rollback teachers along recurrent student-generated histories.

    Each event first follows the stale action for ``r`` solver calls, applies
    CST-R at that same solver state, and then executes the untouched minWM
    suffix under the requested action. Teacher branches are used only to save
    supervision and are never part of inference.
    """
    import torch

    if pipeline.independent_first_frame:
        raise ValueError("Recurrent CST-R capture requires causal chunk history")
    if str(corrector.config.transport_role) != "action_h0":
        raise ValueError("minWM recurrent capture requires a CST-R checkpoint")
    if str(corrector.config.target_parameterization) != "clean_prediction":
        raise ValueError("minWM CST-R must predict the clean requested-action state")

    if noise.ndim != 5:
        raise ValueError("minWM noise must have shape [B,T,C,H,W]")
    batch_size, frame_count, _, _, _ = noise.shape
    chunk_size = int(pipeline.num_frame_per_block)
    denoising_steps = len(pipeline.denoising_step_list)
    if int(corrector.config.latent_channels) != int(noise.shape[2]):
        raise ValueError("Corrector latent channels do not match the minWM state")
    if int(corrector.config.denoising_steps) != denoising_steps:
        raise ValueError("Corrector and minWM denoising schedules differ")
    events = sorted({int(value) for value in event_pose_indices})
    legal_events = set(range(chunk_size, frame_count, chunk_size))
    if not events or not set(events).issubset(legal_events):
        raise ValueError("Events must select later minWM chunk boundaries")
    if set(events) != set(stale_viewmats_by_event):
        raise ValueError("Every event requires exactly one stale camera trajectory")
    receipts = [int(value) for value in receipt_steps]
    if not receipts or any(not 0 < value < denoising_steps for value in receipts):
        raise ValueError("Every receipt must retain at least one minWM cleanup step")

    conditional = pipeline.text_encoder(text_prompts=[text_prompt])
    _reset_pipeline_caches(pipeline, noise)
    output = torch.zeros_like(noise)
    captures: list[TransportCapture] = []
    generator_events: list[dict[str, Any]] = []
    correction_events: list[dict[str, Any]] = []

    def synchronize(tensor: Any) -> None:
        if bool(getattr(tensor, "is_cuda", False)):
            torch.cuda.synchronize(tensor.device)

    def camera_slice(tensor: Any, start: int, count: int) -> Any:
        return tensor[:, start : start + count]

    def model_timestep(value: Any, count: int) -> Any:
        return torch.ones((batch_size, count), device=noise.device, dtype=torch.int64) * value

    def call_generator(
        *,
        active: Any,
        start_frame: int,
        step_index: int | None,
        viewmats: Any,
        phase: str,
    ) -> Any:
        timestep = (
            pipeline.args.context_noise
            if step_index is None
            else pipeline.denoising_step_list[step_index]
        )
        synchronize(active)
        started = time.perf_counter()
        result = pipeline.generator(
            noisy_image_or_video=active,
            conditional_dict=conditional,
            timestep=model_timestep(timestep, int(active.shape[1])),
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=start_frame * pipeline.frame_seq_length,
            viewmats=camera_slice(viewmats, start_frame, int(active.shape[1])),
            Ks=camera_slice(intrinsics, start_frame, int(active.shape[1])),
            prope_kv_cache=pipeline.prope_kv_cache1,
        )
        synchronize(active)
        generator_events.append(
            {
                "start_frame": start_frame,
                "step_index": step_index,
                "phase": phase,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return result

    def transition_noise_bank(active: Any) -> list[Any]:
        return [torch.randn_like(active.flatten(0, 1)) for _ in range(denoising_steps - 1)]

    def add_transition_noise(prediction: Any, transition_noise: Any, next_step: int) -> Any:
        timestep = pipeline.denoising_step_list[next_step]
        count = int(prediction.shape[1])
        return pipeline.scheduler.add_noise(
            prediction.flatten(0, 1),
            transition_noise,
            timestep * torch.ones(batch_size * count, device=prediction.device, dtype=torch.long),
        ).unflatten(0, prediction.shape[:2])

    def run_branch(
        *,
        initial: Any,
        transition_noises: list[Any],
        start_frame: int,
        viewmats: Any,
        branch: str,
    ) -> tuple[list[Any], list[Any]]:
        states = [initial.detach().clone()]
        predictions: list[Any] = []
        active = initial
        for step_index in range(denoising_steps):
            _, prediction = call_generator(
                active=active,
                start_frame=start_frame,
                step_index=step_index,
                viewmats=viewmats,
                phase=f"{branch}_denoise_{step_index + 1}",
            )
            predictions.append(prediction.detach().clone())
            if step_index < denoising_steps - 1:
                active = add_transition_noise(
                    prediction, transition_noises[step_index], step_index + 1
                )
                states.append(active.detach().clone())
        return states, predictions

    with torch.inference_mode():
        for start_frame in range(0, frame_count, chunk_size):
            initial = noise[:, start_frame : start_frame + chunk_size]
            transition_noises = transition_noise_bank(initial)
            if start_frame not in events:
                _, predictions = run_branch(
                    initial=initial,
                    transition_noises=transition_noises,
                    start_frame=start_frame,
                    viewmats=requested_viewmats,
                    branch="uninterrupted",
                )
                student_prediction = predictions[-1]
            else:
                event_ordinal = len(captures)
                stale_viewmats = stale_viewmats_by_event[start_frame]
                history_tail = output[:, start_frame - chunk_size : start_frame].detach().clone()
                old_states, old_predictions = run_branch(
                    initial=initial,
                    transition_noises=transition_noises,
                    start_frame=start_frame,
                    viewmats=stale_viewmats,
                    branch="old_teacher",
                )
                new_states, new_predictions = run_branch(
                    initial=initial,
                    transition_noises=transition_noises,
                    start_frame=start_frame,
                    viewmats=requested_viewmats,
                    branch="rollback_teacher",
                )
                receipt = receipts[event_ordinal % len(receipts)]
                old_camera = camera_slice(stale_viewmats, start_frame, chunk_size)
                new_camera = camera_slice(requested_viewmats, start_frame, chunk_size)
                active_intrinsics = camera_slice(intrinsics, start_frame, chunk_size)
                corrected, correction = apply_transport_model(
                    model=corrector,
                    active_state=old_states[receipt],
                    initial_state=old_states[0],
                    history_tail=history_tail,
                    transition_noises=transition_noises,
                    old_viewmats=old_camera,
                    new_viewmats=new_camera,
                    intrinsics=active_intrinsics,
                    receipt_step=receipt,
                    cached_prediction=old_predictions[receipt - 1],
                    scheduler=pipeline.scheduler,
                    denoising_step_list=pipeline.denoising_step_list,
                    rollout_age=min(
                        event_ordinal, int(getattr(corrector.config, "max_rollout_age", 0))
                    ),
                )
                correction.update(event_ordinal=event_ordinal, start_frame=start_frame)
                correction_events.append(correction)

                active = corrected
                student_prediction = None
                for step_index in range(receipt, denoising_steps):
                    _, student_prediction = call_generator(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        viewmats=requested_viewmats,
                        phase=f"student_cleanup_{step_index + 1}",
                    )
                    if step_index < denoising_steps - 1:
                        active = add_transition_noise(
                            student_prediction,
                            transition_noises[step_index],
                            step_index + 1,
                        )
                if student_prediction is None:
                    raise RuntimeError("CST-R produced no cleanup prediction")

                capture = TransportCapture(
                    tensors={
                        "old_states": torch.stack(old_states),
                        "new_states": torch.stack(new_states),
                        "old_predictions": torch.stack(old_predictions),
                        "new_predictions": torch.stack(new_predictions),
                        "transition_noises": torch.stack(
                            [value.unflatten(0, initial.shape[:2]) for value in transition_noises]
                        ),
                        "history_tail": history_tail,
                        "old_viewmats": old_camera.detach().clone(),
                        "new_viewmats": new_camera.detach().clone(),
                        "intrinsics": active_intrinsics.detach().clone(),
                        "timesteps": pipeline.denoising_step_list.detach().clone(),
                    },
                    metadata={
                        "schema_version": 1,
                        "text_prompt": text_prompt,
                        "event_pose_index": start_frame,
                        "chunk_size": chunk_size,
                        "denoising_steps": denoising_steps,
                        "receipt_steps": list(range(1, denoising_steps)),
                        "teacher_history_source": "student_committed_history",
                        "recurrent_prior_corrections": event_ordinal,
                        "recurrent_event_index": event_ordinal,
                        "recurrent_rollout_age": event_ordinal,
                        "runtime_receipt_step": receipt,
                        "runtime_jump_horizon": 0,
                    },
                )
                validate_transport_capture(capture)
                captures.append(capture)

            output[:, start_frame : start_frame + chunk_size] = student_prediction
            call_generator(
                active=student_prediction,
                start_frame=start_frame,
                step_index=None,
                viewmats=requested_viewmats,
                phase="student_context_commit",
            )

    return (
        captures,
        output,
        {
            "generator_call_count": len(generator_events),
            "generator_elapsed_ms": sum(float(event["elapsed_ms"]) for event in generator_events),
            "correction_count": len(correction_events),
            "correction_elapsed_ms": sum(float(event["elapsed_ms"]) for event in correction_events),
            "generator_events": generator_events,
            "correction_events": correction_events,
        },
    )


__all__ = ["capture_recurrent_cst_r"]
