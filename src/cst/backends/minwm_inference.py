"""Inference-only CST-R/CST-T execution for minWM Wan Action2V."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..core.runtime import apply_transport_model
from .minwm_cache import _reset_pipeline_caches


@dataclass
class MinWMInferenceResult:
    video: Any | None
    latents: Any
    metrics: dict[str, Any]


def run_minwm_cst(
    *,
    pipeline: Any,
    noise: Any,
    text_prompt: str,
    requested_viewmats: Any,
    stale_viewmats_by_event: dict[int, Any],
    intrinsics: Any,
    event_pose_indices: Sequence[int],
    receipt_steps: Sequence[int],
    corrector: Any,
    intra_chunk_offsets_by_event: dict[int, int] | None = None,
    decode: bool = True,
) -> MinWMInferenceResult:
    """Apply one trained corrector per interruption, with no teacher rollout."""
    import torch

    if pipeline.independent_first_frame:
        raise ValueError("minWM CST inference requires causal chunk history")
    role = str(corrector.config.transport_role)
    if role not in {"action_h0", "action_hm"}:
        raise ValueError("Expected a minWM CST-R or CST-T checkpoint")
    method = "cst_t" if role == "action_hm" else "cst_r"
    if str(corrector.config.target_parameterization) != "clean_prediction":
        raise ValueError("minWM correctors must predict clean states")

    batch_size, frame_count, _, _, _ = noise.shape
    chunk_size = int(pipeline.num_frame_per_block)
    denoising_steps = len(pipeline.denoising_step_list)
    if frame_count % chunk_size:
        raise ValueError("Latent frame count must be divisible by the minWM chunk size")
    events = sorted({int(value) for value in event_pose_indices})
    legal_events = set(range(chunk_size, frame_count, chunk_size))
    if not events or not set(events).issubset(legal_events):
        raise ValueError("Events must select later minWM chunk boundaries")
    if set(events) != set(stale_viewmats_by_event):
        raise ValueError("Every event requires exactly one stale camera trajectory")
    receipts = [int(value) for value in receipt_steps]
    if not receipts or any(not 0 < value < denoising_steps for value in receipts):
        raise ValueError("Every receipt must retain at least one cleanup step")
    offsets = {event: int(value) for event, value in (intra_chunk_offsets_by_event or {}).items()}
    if not set(offsets).issubset(events):
        raise ValueError("Intra-chunk offsets must refer to configured events")
    offsets = {event: offsets.get(event, 0) for event in events}
    if method == "cst_r" and any(offsets.values()):
        raise ValueError("CST-R does not use an intra-chunk boundary")
    if method == "cst_t" and any(not 0 < value < chunk_size for value in offsets.values()):
        raise ValueError("CST-T requires a suffix boundary in [1, chunk_size - 1]")

    conditional = pipeline.text_encoder(text_prompts=[text_prompt])
    _reset_pipeline_caches(pipeline, noise)
    output = torch.zeros_like(noise)
    generator_events: list[dict[str, Any]] = []
    correction_events: list[dict[str, Any]] = []
    interruption_events: list[dict[str, Any]] = []
    wall_origin = time.perf_counter()

    def synchronize(tensor: Any) -> None:
        if bool(getattr(tensor, "is_cuda", False)):
            torch.cuda.synchronize(tensor.device)

    def elapsed_ms() -> float:
        return (time.perf_counter() - wall_origin) * 1000.0

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

    with torch.inference_mode():
        for start_frame in range(0, frame_count, chunk_size):
            initial = noise[:, start_frame : start_frame + chunk_size]
            transition_noises = transition_noise_bank(initial)
            if start_frame not in events:
                active = initial
                prediction = None
                for step_index in range(denoising_steps):
                    _, prediction = call_generator(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        viewmats=requested_viewmats,
                        phase=f"uninterrupted_denoise_{step_index + 1}",
                    )
                    if step_index < denoising_steps - 1:
                        active = add_transition_noise(
                            prediction, transition_noises[step_index], step_index + 1
                        )
                if prediction is None:
                    raise RuntimeError("Uninterrupted chunk produced no prediction")
            else:
                event_ordinal = len(interruption_events)
                receipt = receipts[event_ordinal % len(receipts)]
                boundary = offsets[start_frame]
                stale_viewmats = stale_viewmats_by_event[start_frame]
                active = initial
                prefix_prediction = None
                saved_states = [initial]
                for step_index in range(receipt):
                    _, prefix_prediction = call_generator(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        viewmats=stale_viewmats,
                        phase=f"{method}_stale_prefix_{step_index + 1}",
                    )
                    active = add_transition_noise(
                        prefix_prediction, transition_noises[step_index], step_index + 1
                    )
                    saved_states.append(active)
                if prefix_prediction is None:
                    raise RuntimeError("The stale prefix produced no prediction")
                request_ms = elapsed_ms()

                history_tail = output[:, start_frame - chunk_size : start_frame].detach().clone()
                mask = None
                if boundary:
                    mask = torch.zeros(
                        [batch_size, chunk_size, 1, 1, 1],
                        device=noise.device,
                        dtype=noise.dtype,
                    )
                    mask[:, boundary:] = 1.0
                corrected, correction = apply_transport_model(
                    model=corrector,
                    active_state=active,
                    initial_state=initial,
                    history_tail=history_tail,
                    transition_noises=transition_noises,
                    old_viewmats=camera_slice(stale_viewmats, start_frame, chunk_size),
                    new_viewmats=camera_slice(requested_viewmats, start_frame, chunk_size),
                    intrinsics=camera_slice(intrinsics, start_frame, chunk_size),
                    receipt_step=receipt,
                    cached_prediction=prefix_prediction,
                    scheduler=pipeline.scheduler,
                    denoising_step_list=pipeline.denoising_step_list,
                    rollout_age=min(
                        event_ordinal, int(getattr(corrector.config, "max_rollout_age", 0))
                    ),
                    temporal_suffix_mask=mask,
                )
                if boundary:
                    corrected = torch.cat(
                        [saved_states[receipt][:, :boundary], corrected[:, boundary:]], dim=1
                    )
                correction.update(event_ordinal=event_ordinal, start_frame=start_frame)
                correction_events.append(correction)
                transport_ready_ms = elapsed_ms()

                active = corrected
                prediction = None
                cleanup_calls = 0
                for step_index in range(receipt, denoising_steps):
                    _, prediction = call_generator(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        viewmats=requested_viewmats,
                        phase=f"{method}_cleanup_{step_index + 1}",
                    )
                    cleanup_calls += 1
                    if boundary:
                        prediction = torch.cat(
                            [prefix_prediction[:, :boundary], prediction[:, boundary:]], dim=1
                        )
                    if step_index < denoising_steps - 1:
                        active = add_transition_noise(
                            prediction, transition_noises[step_index], step_index + 1
                        )
                if prediction is None:
                    raise RuntimeError("CST cleanup produced no prediction")
                ready_ms = elapsed_ms()
                interruption_events.append(
                    {
                        "event_index": event_ordinal,
                        "event_pose_index": start_frame,
                        "receipt_step": receipt,
                        "intra_chunk_offset": boundary,
                        "method": method,
                        "transport_latency_ms": transport_ready_ms - request_ms,
                        "response_latency_ms": ready_ms - request_ms,
                        "post_request_backbone_nfe": cleanup_calls,
                        "continuation_semantics": "same_step_correct_then_resume",
                    }
                )

            output[:, start_frame : start_frame + chunk_size] = prediction
            call_generator(
                active=prediction,
                start_frame=start_frame,
                step_index=None,
                viewmats=requested_viewmats,
                phase="context_commit",
            )

    video = None
    decode_ms = 0.0
    if decode:
        synchronize(output)
        started = time.perf_counter()
        video = pipeline.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        synchronize(output)
        decode_ms = (time.perf_counter() - started) * 1000.0

    return MinWMInferenceResult(
        video=video,
        latents=output,
        metrics={
            "method": method,
            "interruption_count": len(interruption_events),
            "generator_call_count": len(generator_events),
            "generator_elapsed_ms": sum(float(event["elapsed_ms"]) for event in generator_events),
            "correction_elapsed_ms": sum(float(event["elapsed_ms"]) for event in correction_events),
            "decode_elapsed_ms": decode_ms,
            "rollout_wall_ms": elapsed_ms(),
            "interruption_events": interruption_events,
            "generator_events": generator_events,
            "correction_events": correction_events,
        },
    )


__all__ = ["MinWMInferenceResult", "run_minwm_cst"]
