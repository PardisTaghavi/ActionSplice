"""Inference-only CST-R/CST-T policies for official HY-WorldPlay."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.runtime import apply_transport_model
from ..core.state import HY15_ACTION2V_STATE_SPEC
from .hyworld15 import (
    HYWorldPlayCondition,
    _build_context_caches,
    _call_flow_model,
    _condition_to_device,
    _euler_step,
    _initialize_text_caches,
    _scheduler_sigmas,
)


@dataclass
class HYWorldPlayInferenceResult:
    output: Any
    metrics: dict[str, Any]


def run_hyworld15_cst(
    *,
    pipeline: Any,
    latents: Any,
    timesteps: Any,
    prompt_embeds: Any,
    prompt_mask: Any,
    vision_states: Any,
    cond_latents: Any,
    task_type: str,
    extra_kwargs: dict[str, Any],
    requested_condition: HYWorldPlayCondition,
    stale_conditions_by_event: dict[int, HYWorldPlayCondition],
    event_pose_indices: Sequence[int],
    receipt_steps: Sequence[int],
    corrector: Any,
    intra_chunk_offsets_by_event: dict[int, int] | None = None,
    history_selector: Callable[..., Sequence[int]] | None = None,
) -> HYWorldPlayInferenceResult:
    """Apply trained HY CST correctors without executing teacher branches."""
    import torch

    if latents.ndim != 5:
        raise ValueError("HY-WorldPlay latents must have shape [B,C,T,H,W]")
    batch_size, channels, frame_count, _, _ = latents.shape
    spec = HY15_ACTION2V_STATE_SPEC
    chunk_size = int(getattr(pipeline, "chunk_latent_frames", 4))
    denoising_steps = int(timesteps.numel())
    if channels != spec.latent_channels:
        raise ValueError(f"Expected {spec.latent_channels} HY latent channels")
    if chunk_size != 4 or denoising_steps != spec.denoising_steps:
        raise ValueError("HY-WM1.5 CST requires the official 4x4 sampler")
    if frame_count % chunk_size:
        raise ValueError("Latent frame count must be divisible by four")

    role = str(corrector.config.transport_role)
    if role not in {"action_h0_state", "action_hm_state"}:
        raise ValueError("Expected an HY-WM1.5 CST-R or CST-T checkpoint")
    method = "cst_t" if role == "action_hm_state" else "cst_r"
    if str(corrector.config.target_parameterization) != "state":
        raise ValueError("HY-WM1.5 correctors must predict direct Euler states")
    if int(corrector.config.latent_channels) != spec.latent_channels:
        raise ValueError("Corrector latent channels do not match HY-WM1.5")
    if int(corrector.config.denoising_steps) != denoising_steps:
        raise ValueError("Corrector and sampler denoising schedules differ")

    events = sorted({int(value) for value in event_pose_indices})
    legal_events = set(range(chunk_size, frame_count, chunk_size))
    if not events or not set(events).issubset(legal_events):
        raise ValueError("Events must select later HY chunk boundaries")
    if set(events) != set(stale_conditions_by_event):
        raise ValueError("Every event requires exactly one stale condition")
    receipts = [int(value) for value in receipt_steps]
    if not receipts or any(not 0 < value < denoising_steps for value in receipts):
        raise ValueError("Every receipt must retain at least one HY cleanup step")
    offsets = {event: int(value) for event, value in (intra_chunk_offsets_by_event or {}).items()}
    if not set(offsets).issubset(events):
        raise ValueError("Intra-chunk offsets must refer to configured events")
    offsets = {event: offsets.get(event, 0) for event in events}
    if method == "cst_r" and any(offsets.values()):
        raise ValueError("CST-R does not use an intra-chunk boundary")
    if method == "cst_t" and any(not 0 < value < chunk_size for value in offsets.values()):
        raise ValueError("CST-T requires a suffix boundary in [1, 3]")

    requested_condition.validate(batch_size=batch_size, frame_count=frame_count)
    requested_condition = _condition_to_device(requested_condition, latents.device)
    stale_conditions_by_event = {
        event: _condition_to_device(condition, latents.device)
        for event, condition in stale_conditions_by_event.items()
    }
    for condition in stale_conditions_by_event.values():
        condition.validate(batch_size=batch_size, frame_count=frame_count)
    if history_selector is None:
        from hyvideo.utils.retrieval_context import select_aligned_memory_frames

        history_selector = select_aligned_memory_frames

    def synchronize() -> None:
        if latents.is_cuda:
            torch.cuda.synchronize(latents.device)

    synchronize()
    wall_origin = time.perf_counter()
    positive_cache, negative_cache = _initialize_text_caches(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        vision_states=vision_states,
        task_type=task_type,
        extra_kwargs=extra_kwargs,
        device=latents.device,
    )
    sigmas = _scheduler_sigmas(pipeline.scheduler, timesteps)
    output = latents.detach().clone()
    generator_events: list[dict[str, Any]] = []
    correction_events: list[dict[str, Any]] = []
    context_events: list[dict[str, Any]] = []
    interruption_events: list[dict[str, Any]] = []

    def elapsed_ms() -> float:
        return (time.perf_counter() - wall_origin) * 1000.0

    def build_context(
        start_frame: int, condition: HYWorldPlayCondition, phase: str
    ) -> tuple[tuple[list[dict[str, Any]], list[dict[str, Any]] | None], list[int]]:
        synchronize()
        started = time.perf_counter()
        caches, selected = _build_context_caches(
            pipeline=pipeline,
            output=output,
            cond_latents=cond_latents,
            start_frame=start_frame,
            condition=condition,
            task_type=task_type,
            timesteps=timesteps,
            positive_text_cache=positive_cache,
            negative_text_cache=negative_cache,
            history_selector=history_selector,
        )
        synchronize()
        context_events.append(
            {
                "phase": phase,
                "start_frame": start_frame,
                "selected_memory_frames": selected,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return caches, selected

    def flow(
        *,
        active: Any,
        start_frame: int,
        step_index: int,
        condition: HYWorldPlayCondition,
        caches: tuple[list[dict[str, Any]], list[dict[str, Any]] | None],
        selected: Sequence[int],
        phase: str,
    ) -> Any:
        return _call_flow_model(
            pipeline=pipeline,
            active=active,
            cond_latents=cond_latents,
            start_frame=start_frame,
            timestep=timesteps[step_index],
            condition=condition,
            caches=caches,
            selected_count=len(selected),
            task_type=task_type,
            branch=phase,
            step_index=step_index,
            events=generator_events,
        )

    with torch.inference_mode():
        for start_frame in range(0, frame_count, chunk_size):
            initial = latents[:, :, start_frame : start_frame + chunk_size]
            if start_frame not in events:
                caches, selected = build_context(
                    start_frame, requested_condition, "committed_context"
                )
                active = initial
                for step_index in range(denoising_steps):
                    prediction = flow(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        condition=requested_condition,
                        caches=caches,
                        selected=selected,
                        phase="uninterrupted",
                    )
                    active = _euler_step(active, prediction, sigmas, step_index)
                output[:, :, start_frame : start_frame + chunk_size] = active.to(output.dtype)
                continue

            event_ordinal = len(interruption_events)
            receipt = receipts[event_ordinal % len(receipts)]
            boundary = offsets[start_frame]
            stale_condition = stale_conditions_by_event[start_frame]
            stale_caches, stale_selected = build_context(
                start_frame, stale_condition, "stale_pre_request_context"
            )
            active = initial
            for step_index in range(receipt):
                prediction = flow(
                    active=active,
                    start_frame=start_frame,
                    step_index=step_index,
                    condition=stale_condition,
                    caches=stale_caches,
                    selected=stale_selected,
                    phase=f"{method}_stale_prefix",
                )
                active = _euler_step(active, prediction, sigmas, step_index)
            request_ms = elapsed_ms()

            canonical_active = spec.to_canonical(active)
            history = output[:, :, start_frame - chunk_size : start_frame].detach().clone()
            mask = None
            if boundary:
                mask = torch.zeros(
                    [batch_size, chunk_size, 1, 1, 1],
                    device=canonical_active.device,
                    dtype=canonical_active.dtype,
                )
                mask[:, boundary:] = 1.0
            zero_trace = [
                torch.zeros_like(canonical_active.flatten(0, 1)) for _ in range(denoising_steps - 1)
            ]
            corrected, correction = apply_transport_model(
                model=corrector,
                active_state=canonical_active,
                initial_state=spec.to_canonical(initial),
                history_tail=spec.to_canonical(history),
                transition_noises=zero_trace,
                old_viewmats=stale_condition.viewmats[:, start_frame : start_frame + chunk_size],
                new_viewmats=requested_condition.viewmats[
                    :, start_frame : start_frame + chunk_size
                ],
                intrinsics=requested_condition.intrinsics[
                    :, start_frame : start_frame + chunk_size
                ],
                receipt_step=receipt,
                cached_prediction=canonical_active,
                rollout_age=min(
                    event_ordinal, int(getattr(corrector.config, "max_rollout_age", 0))
                ),
                temporal_suffix_mask=mask,
            )
            correction.update(event_ordinal=event_ordinal, start_frame=start_frame)
            correction_events.append(correction)
            transport_ready_ms = elapsed_ms()

            requested_caches, requested_selected = build_context(
                start_frame, requested_condition, "requested_cleanup_context"
            )
            active = spec.from_canonical(corrected)
            cleanup_calls = 0
            for step_index in range(receipt, denoising_steps):
                prediction = flow(
                    active=active,
                    start_frame=start_frame,
                    step_index=step_index,
                    condition=requested_condition,
                    caches=requested_caches,
                    selected=requested_selected,
                    phase=f"{method}_cleanup",
                )
                active = _euler_step(active, prediction, sigmas, step_index)
                cleanup_calls += 1
            output[:, :, start_frame : start_frame + chunk_size] = active.to(output.dtype)
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

    synchronize()
    return HYWorldPlayInferenceResult(
        output=output,
        metrics={
            "method": method,
            "interruption_count": len(interruption_events),
            "generator_call_count": len(generator_events),
            "generator_elapsed_ms": sum(float(event["elapsed_ms"]) for event in generator_events),
            "context_elapsed_ms": sum(float(event["elapsed_ms"]) for event in context_events),
            "correction_elapsed_ms": sum(float(event["elapsed_ms"]) for event in correction_events),
            "rollout_wall_ms": elapsed_ms(),
            "interruption_events": interruption_events,
            "generator_events": generator_events,
            "context_events": context_events,
            "correction_events": correction_events,
        },
    )


class OfficialHYWorldPlayInferenceHook(AbstractContextManager["OfficialHYWorldPlayInferenceHook"]):
    """Install one inference-only CST policy at HY-WorldPlay's AR boundary."""

    def __init__(
        self,
        *,
        pipeline: Any,
        stale_conditions_by_event: dict[int, HYWorldPlayCondition],
        event_pose_indices: Sequence[int],
        receipt_steps: Sequence[int],
        corrector: Any,
        intra_chunk_offsets_by_event: dict[int, int] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.stale_conditions_by_event = stale_conditions_by_event
        self.event_pose_indices = list(event_pose_indices)
        self.receipt_steps = list(receipt_steps)
        self.corrector = corrector
        self.intra_chunk_offsets_by_event = dict(intra_chunk_offsets_by_event or {})
        self.original_ar_rollout: Any | None = None
        self.result: HYWorldPlayInferenceResult | None = None

    def __enter__(self) -> "OfficialHYWorldPlayInferenceHook":
        self.original_ar_rollout = self.pipeline.ar_rollout

        def wrapped_ar_rollout(**kwargs: Any) -> Any:
            requested = HYWorldPlayCondition(
                viewmats=kwargs["viewmats"],
                intrinsics=kwargs["Ks"],
                actions=kwargs["action"],
            )
            context = nullcontext()
            if bool(getattr(self.pipeline, "enable_offloading", False)):
                from hyvideo.commons import auto_offload_model

                context = auto_offload_model(
                    self.pipeline.transformer,
                    self.pipeline.execution_device,
                    enabled=True,
                )
            with context:
                self.result = run_hyworld15_cst(
                    pipeline=self.pipeline,
                    latents=kwargs["latents"],
                    timesteps=kwargs["timesteps"],
                    prompt_embeds=kwargs["prompt_embeds"],
                    prompt_mask=kwargs["prompt_mask"],
                    vision_states=kwargs["vision_states"],
                    cond_latents=kwargs["cond_latents"],
                    task_type=kwargs["task_type"],
                    extra_kwargs=kwargs["extra_kwargs"],
                    requested_condition=requested,
                    stale_conditions_by_event=self.stale_conditions_by_event,
                    event_pose_indices=self.event_pose_indices,
                    receipt_steps=self.receipt_steps,
                    corrector=self.corrector,
                    intra_chunk_offsets_by_event=self.intra_chunk_offsets_by_event,
                )
            return self.result.output

        self.pipeline.ar_rollout = wrapped_ar_rollout
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.original_ar_rollout is not None:
            self.pipeline.ar_rollout = self.original_ar_rollout
        return None


__all__ = [
    "HYWorldPlayInferenceResult",
    "OfficialHYWorldPlayInferenceHook",
    "run_hyworld15_cst",
]
