"""Interruption baselines for the official HY-World 1.5 AR sampler."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .hyworld15 import (
    HYWorldPlayCondition,
    _build_context_caches,
    _call_flow_model,
    _condition_to_device,
    _euler_step,
    _initialize_text_caches,
    _scheduler_sigmas,
)

BASELINE_POLICIES = frozenset({"full_rollback", "wait"})


@dataclass
class HYWorldPlayBaselineResult:
    """Generated native latents and detailed latency accounting."""

    output: Any
    metrics: dict[str, Any]


def run_hyworld15_interruption_baseline(
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
    wait_condition: HYWorldPlayCondition | None,
    stale_conditions_by_event: dict[int, HYWorldPlayCondition],
    event_pose_indices: Sequence[int],
    runtime_receipt_steps: Sequence[int],
    policy: str,
    history_selector: Callable[..., Sequence[int]] | None = None,
) -> HYWorldPlayBaselineResult:
    """Run matched full-rollback or wait behavior at multiple interruptions.

    A receipt value ``r`` means the request arrives after exactly ``r`` of the
    four stale-action denoising evaluations have completed.

    ``full_rollback`` discards those evaluations, restores the chunk's initial
    noise, rebuilds the requested-action context, and runs all four requested
    evaluations. ``wait`` completes the active stale-action chunk and applies
    the request to the following chunk.
    """
    import torch

    if policy not in BASELINE_POLICIES:
        raise ValueError(f"Unknown HY-World baseline policy: {policy!r}")
    if latents.ndim != 5:
        raise ValueError("HY-WorldPlay latents must have shape [B,C,T,H,W]")
    batch_size, _, frame_count, _, _ = latents.shape
    chunk_size = int(getattr(pipeline, "chunk_latent_frames", 4))
    denoising_steps = int(timesteps.numel())
    if chunk_size != 4 or denoising_steps != 4:
        raise ValueError("HY-World 1.5 baselines require the official 4x4 AR sampler")
    if frame_count % chunk_size:
        raise ValueError("Latent frame count must be divisible by four")
    if not runtime_receipt_steps or any(
        int(receipt) <= 0 or int(receipt) >= denoising_steps for receipt in runtime_receipt_steps
    ):
        raise ValueError("Receipt steps must be in [1, denoising_steps - 1]")

    events = sorted({int(value) for value in event_pose_indices})
    legal_events = set(range(chunk_size, frame_count, chunk_size))
    if not events or not set(events).issubset(legal_events):
        raise ValueError("Interruptions must occur at later HY chunk boundaries")
    if policy == "full_rollback" and set(events) != set(stale_conditions_by_event):
        raise ValueError("Full rollback requires one stale condition per event")
    if policy == "wait":
        if wait_condition is None:
            raise ValueError("Wait requires a delayed-action condition")
        if any(event + chunk_size in events for event in events):
            raise ValueError("Wait events need one non-event response chunk between them")
        if events[-1] + chunk_size >= frame_count:
            raise ValueError("Wait needs a response chunk after the final interruption")

    requested_condition.validate(batch_size=batch_size, frame_count=frame_count)
    requested_condition = _condition_to_device(requested_condition, latents.device)
    if wait_condition is not None:
        wait_condition.validate(batch_size=batch_size, frame_count=frame_count)
        wait_condition = _condition_to_device(wait_condition, latents.device)
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
    rollout_started = time.perf_counter()
    text_started = time.perf_counter()
    positive_cache, negative_cache = _initialize_text_caches(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        vision_states=vision_states,
        task_type=task_type,
        extra_kwargs=extra_kwargs,
        device=latents.device,
    )
    synchronize()
    text_cache_elapsed_ms = (time.perf_counter() - text_started) * 1000.0

    sigmas = _scheduler_sigmas(pipeline.scheduler, timesteps)
    output = latents.detach().clone()
    generator_events: list[dict[str, Any]] = []
    context_events: list[dict[str, Any]] = []
    interruption_events: list[dict[str, Any]] = []
    pending_wait_event: dict[str, Any] | None = None

    def elapsed_ms() -> float:
        return (time.perf_counter() - rollout_started) * 1000.0

    def build_context(
        *,
        start_frame: int,
        condition: HYWorldPlayCondition,
        phase: str,
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

    def denoise_step(
        *,
        active: Any,
        start_frame: int,
        step_index: int,
        condition: HYWorldPlayCondition,
        caches: tuple[list[dict[str, Any]], list[dict[str, Any]] | None],
        selected: Sequence[int],
        branch: str,
    ) -> Any:
        flow = _call_flow_model(
            pipeline=pipeline,
            active=active,
            cond_latents=cond_latents,
            start_frame=start_frame,
            timestep=timesteps[step_index],
            condition=condition,
            caches=caches,
            selected_count=len(selected),
            task_type=task_type,
            branch=branch,
            step_index=step_index,
            events=generator_events,
        )
        return _euler_step(active, flow, sigmas, step_index)

    def full_chunk(
        *,
        initial: Any,
        start_frame: int,
        condition: HYWorldPlayCondition,
        branch: str,
        context_phase: str,
    ) -> Any:
        caches, selected = build_context(
            start_frame=start_frame,
            condition=condition,
            phase=context_phase,
        )
        active = initial
        for step_index in range(denoising_steps):
            active = denoise_step(
                active=active,
                start_frame=start_frame,
                step_index=step_index,
                condition=condition,
                caches=caches,
                selected=selected,
                branch=branch,
            )
        return active

    with torch.inference_mode():
        for start_frame in range(0, frame_count, chunk_size):
            initial = latents[:, :, start_frame : start_frame + chunk_size]
            if start_frame not in events:
                condition = requested_condition if policy == "full_rollback" else wait_condition
                if condition is None:  # pragma: no cover - validated above
                    raise RuntimeError("Missing committed condition")
                final = full_chunk(
                    initial=initial,
                    start_frame=start_frame,
                    condition=condition,
                    branch="uninterrupted",
                    context_phase="committed_context",
                )
                output[:, :, start_frame : start_frame + chunk_size] = final.to(output.dtype)
                if pending_wait_event is not None:
                    synchronize()
                    ready = elapsed_ms()
                    pending_wait_event.update(
                        {
                            "response_chunk_start_frame": start_frame,
                            "response_action_labels": _action_labels(
                                condition, start_frame, chunk_size
                            ),
                            "response_ready_wall_ms": ready,
                            "response_latency_ms": (ready - pending_wait_event["request_wall_ms"]),
                            "post_request_nfe_to_response": (
                                denoising_steps
                                - pending_wait_event["receipt_step"]
                                + denoising_steps
                            ),
                        }
                    )
                    pending_wait_event = None
                continue

            event_ordinal = len(interruption_events)
            receipt = int(runtime_receipt_steps[event_ordinal % len(runtime_receipt_steps)])
            event_record: dict[str, Any] = {
                "event_index": event_ordinal,
                "event_pose_index": start_frame,
                "receipt_step": receipt,
                "policy": policy,
                "requested_action_labels": _action_labels(
                    requested_condition, start_frame, chunk_size
                ),
            }

            if policy == "full_rollback":
                stale_condition = stale_conditions_by_event[start_frame]
                stale_caches, stale_selected = build_context(
                    start_frame=start_frame,
                    condition=stale_condition,
                    phase="stale_pre_request_context",
                )
                active = initial
                for step_index in range(receipt):
                    active = denoise_step(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        condition=stale_condition,
                        caches=stale_caches,
                        selected=stale_selected,
                        branch="rollback_discarded_prefix",
                    )
                synchronize()
                request_wall_ms = elapsed_ms()
                event_record.update(
                    {
                        "stale_action_labels": _action_labels(
                            stale_condition, start_frame, chunk_size
                        ),
                        "request_wall_ms": request_wall_ms,
                        "discarded_nfe": receipt,
                        "total_event_nfe": receipt + denoising_steps,
                        "post_request_nfe_to_response": denoising_steps,
                    }
                )
                final = full_chunk(
                    initial=initial,
                    start_frame=start_frame,
                    condition=requested_condition,
                    branch="rollback_requested_restart",
                    context_phase="requested_post_request_context",
                )
                output[:, :, start_frame : start_frame + chunk_size] = final.to(output.dtype)
                synchronize()
                ready = elapsed_ms()
                event_record.update(
                    {
                        "response_chunk_start_frame": start_frame,
                        "current_chunk_ready_wall_ms": ready,
                        "current_chunk_post_request_latency_ms": (ready - request_wall_ms),
                        "response_ready_wall_ms": ready,
                        "response_latency_ms": ready - request_wall_ms,
                    }
                )
            else:
                if wait_condition is None:  # pragma: no cover - validated above
                    raise RuntimeError("Missing wait condition")
                caches, selected = build_context(
                    start_frame=start_frame,
                    condition=wait_condition,
                    phase="wait_stale_context",
                )
                active = initial
                request_wall_ms = 0.0
                for step_index in range(denoising_steps):
                    active = denoise_step(
                        active=active,
                        start_frame=start_frame,
                        step_index=step_index,
                        condition=wait_condition,
                        caches=caches,
                        selected=selected,
                        branch="wait_complete_stale_chunk",
                    )
                    if step_index + 1 == receipt:
                        synchronize()
                        request_wall_ms = elapsed_ms()
                output[:, :, start_frame : start_frame + chunk_size] = active.to(output.dtype)
                synchronize()
                current_ready = elapsed_ms()
                event_record.update(
                    {
                        "stale_action_labels": _action_labels(
                            wait_condition, start_frame, chunk_size
                        ),
                        "request_wall_ms": request_wall_ms,
                        "discarded_nfe": 0,
                        "total_event_nfe": denoising_steps,
                        "current_chunk_ready_wall_ms": current_ready,
                        "current_chunk_post_request_latency_ms": (current_ready - request_wall_ms),
                    }
                )
                pending_wait_event = event_record

            interruption_events.append(event_record)

    if pending_wait_event is not None:
        raise RuntimeError("Wait rollout ended before the requested response chunk")
    synchronize()
    rollout_wall_ms = elapsed_ms()
    generator_elapsed_ms = sum(float(event["elapsed_ms"]) for event in generator_events)
    context_elapsed_ms = sum(float(event["elapsed_ms"]) for event in context_events)
    guidance_multiplier = 2 if pipeline.do_classifier_free_guidance else 1
    context_forward_count = sum(
        guidance_multiplier for event in context_events if event["selected_memory_frames"]
    )
    return HYWorldPlayBaselineResult(
        output=output,
        metrics={
            "policy": policy,
            "interruption_count": len(interruption_events),
            "denoising_steps": denoising_steps,
            "chunk_size": chunk_size,
            "latent_frame_count": frame_count,
            "generator_call_count": len(generator_events),
            "generator_transformer_forward_count": (len(generator_events) * guidance_multiplier),
            "context_transformer_forward_count": context_forward_count,
            "text_transformer_forward_count": guidance_multiplier,
            "discarded_nfe": sum(int(event["discarded_nfe"]) for event in interruption_events),
            "generator_elapsed_ms": generator_elapsed_ms,
            "context_elapsed_ms": context_elapsed_ms,
            "text_cache_elapsed_ms": text_cache_elapsed_ms,
            "rollout_wall_ms": rollout_wall_ms,
            "rollout_accounting_overhead_ms": max(
                0.0,
                rollout_wall_ms - generator_elapsed_ms - context_elapsed_ms - text_cache_elapsed_ms,
            ),
            "interruption_events": interruption_events,
            "generator_events": generator_events,
            "context_events": context_events,
        },
    )


class OfficialHYWorldPlayBaselineHook(AbstractContextManager["OfficialHYWorldPlayBaselineHook"]):
    """Install a baseline AR loop without modifying the upstream checkout."""

    def __init__(
        self,
        *,
        pipeline: Any,
        wait_condition: HYWorldPlayCondition | None,
        stale_conditions_by_event: dict[int, HYWorldPlayCondition],
        event_pose_indices: Sequence[int],
        runtime_receipt_steps: Sequence[int],
        policy: str,
    ) -> None:
        self.pipeline = pipeline
        self.wait_condition = wait_condition
        self.stale_conditions_by_event = stale_conditions_by_event
        self.event_pose_indices = list(event_pose_indices)
        self.runtime_receipt_steps = list(runtime_receipt_steps)
        self.policy = policy
        self.original_ar_rollout: Any | None = None
        self.result: HYWorldPlayBaselineResult | None = None

    def __enter__(self) -> "OfficialHYWorldPlayBaselineHook":
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
                self.result = run_hyworld15_interruption_baseline(
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
                    wait_condition=self.wait_condition,
                    stale_conditions_by_event=self.stale_conditions_by_event,
                    event_pose_indices=self.event_pose_indices,
                    runtime_receipt_steps=self.runtime_receipt_steps,
                    policy=self.policy,
                )
            return self.result.output

        self.pipeline.ar_rollout = wrapped_ar_rollout
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.original_ar_rollout is not None:
            self.pipeline.ar_rollout = self.original_ar_rollout
        return None


def _action_labels(
    condition: HYWorldPlayCondition,
    start_frame: int,
    chunk_size: int,
) -> list[int]:
    labels = condition.actions[0, start_frame : start_frame + chunk_size]
    return [int(value) for value in labels.detach().cpu().tolist()]


__all__ = [
    "BASELINE_POLICIES",
    "HYWorldPlayBaselineResult",
    "OfficialHYWorldPlayBaselineHook",
    "run_hyworld15_interruption_baseline",
]
