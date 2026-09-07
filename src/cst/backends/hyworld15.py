"""External CST extension for the official HY-World 1.5 inference stack.

The upstream checkout remains unmodified.  This module replaces only the
``ar_rollout`` call boundary, where all prompt/image conditioning has already
been prepared, and executes the same four-step autoregressive chunk sampler
with matched old/new branches and recurrent student-history capture.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.state import HY15_ACTION2V_STATE_SPEC
from ..data.capture import TransportCapture, validate_transport_capture


@dataclass(frozen=True)
class HYWorldPlayCondition:
    """Full latent-frame camera and discrete-action trajectory."""

    viewmats: Any
    intrinsics: Any
    actions: Any

    def validate(self, *, batch_size: int, frame_count: int) -> None:
        expected = (
            ("viewmats", self.viewmats, (4, 4)),
            ("intrinsics", self.intrinsics, (3, 3)),
        )
        for name, tensor, tail in expected:
            if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != tail:
                raise ValueError(f"{name} must have shape [B,T,{tail[0]},{tail[1]}]")
            if int(tensor.shape[0]) != batch_size or int(tensor.shape[1]) < frame_count:
                raise ValueError(f"{name} does not cover the latent rollout")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape [B,T]")
        if int(self.actions.shape[0]) != batch_size or int(self.actions.shape[1]) < frame_count:
            raise ValueError("actions do not cover the latent rollout")


@dataclass
class HYWorldPlayRecurrentResult:
    captures: list[TransportCapture]
    output: Any
    metrics: dict[str, Any]


def capture_hyworld15_recurrent_sequence(
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
    runtime_receipt_steps: Sequence[int],
    transport_model: Any | None,
    intra_chunk_offsets_by_event: dict[int, int] | None = None,
    history_selector: Callable[..., Sequence[int]] | None = None,
) -> HYWorldPlayRecurrentResult:
    """Run official HY-WorldPlay with repeated CST-R interruptions.

    HY-WorldPlay uses deterministic Euler flow updates.  Its corrector must
    therefore be configured with ``target_parameterization='state'``.  The
    target is the exact matched new-action sampler state at the receipt step;
    the untouched official suffix then performs at least one cleanup NFE.
    """
    import torch

    spec = HY15_ACTION2V_STATE_SPEC
    if latents.ndim != 5:
        raise ValueError("HY-WorldPlay latents must have shape [B,C,T,H,W]")
    batch_size, channels, frame_count, _, _ = latents.shape
    chunk_size = int(getattr(pipeline, "chunk_latent_frames", 4))
    denoising_steps = int(timesteps.numel())
    if channels != spec.latent_channels:
        raise ValueError(f"Expected {spec.latent_channels} HY latent channels")
    if chunk_size != 4 or denoising_steps != spec.denoising_steps:
        raise ValueError("Official HY-World 1.5 CST currently targets 4x4 AR inference")
    if frame_count % chunk_size:
        raise ValueError("Latent frame count must be divisible by four")
    if not runtime_receipt_steps:
        raise ValueError("runtime_receipt_steps cannot be empty")
    if any(not 0 < int(value) < denoising_steps for value in runtime_receipt_steps):
        raise ValueError("Every receipt must retain an official HY cleanup step")
    events = sorted({int(value) for value in event_pose_indices})
    legal_events = set(range(chunk_size, frame_count, chunk_size))
    if not events or not set(events).issubset(legal_events):
        raise ValueError("Events must be later HY chunk boundaries")
    if set(events) != set(stale_conditions_by_event):
        raise ValueError("Every event requires exactly one stale trajectory")
    offsets = {event: int(value) for event, value in (intra_chunk_offsets_by_event or {}).items()}
    if not set(offsets).issubset(events):
        raise ValueError("Intra-chunk offsets must refer to configured events")
    offsets = {event: offsets.get(event, 0) for event in events}
    if any(not 0 <= value < chunk_size for value in offsets.values()):
        raise ValueError("Every intra-chunk offset must be in [0, chunk_size)")
    requested_condition.validate(batch_size=batch_size, frame_count=frame_count)
    for condition in stale_conditions_by_event.values():
        condition.validate(batch_size=batch_size, frame_count=frame_count)
    requested_condition = _condition_to_device(requested_condition, latents.device)
    stale_conditions_by_event = {
        event: _condition_to_device(condition, latents.device)
        for event, condition in stale_conditions_by_event.items()
    }

    model_config = None if transport_model is None else transport_model.config
    if model_config is not None:
        if str(model_config.target_parameterization) != "state":
            raise ValueError("Official HY-World 1.5 requires direct state transport")
        if int(model_config.latent_channels) != spec.latent_channels:
            raise ValueError("Corrector latent channels do not match HY-World 1.5")
        if int(model_config.denoising_steps) != denoising_steps:
            raise ValueError("Corrector denoising schedule does not match HY-World 1.5")
        if int(model_config.max_jump_horizon) != 0:
            raise ValueError("Official HY CST-R/CST-T checkpoints must be same-step")
        expected_role = "action_hm_state" if any(offsets.values()) else "action_h0_state"
        if str(model_config.transport_role) != expected_role:
            raise ValueError(f"Capture boundaries require checkpoint role {expected_role!r}")

    if history_selector is None:
        from hyvideo.utils.retrieval_context import select_aligned_memory_frames

        history_selector = select_aligned_memory_frames

    sigmas = _scheduler_sigmas(pipeline.scheduler, timesteps)
    output = latents.detach().clone()
    positive_cache, negative_cache = _initialize_text_caches(
        pipeline=pipeline,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        vision_states=vision_states,
        task_type=task_type,
        extra_kwargs=extra_kwargs,
        device=latents.device,
    )
    captures: list[TransportCapture] = []
    generator_events: list[dict[str, Any]] = []
    correction_events: list[dict[str, Any]] = []

    def run_branch(
        *,
        initial: Any,
        start_frame: int,
        condition: HYWorldPlayCondition,
        branch: str,
        prefix_reference_states: Sequence[Any] | None = None,
        prefix_reference_flows: Sequence[Any] | None = None,
        prefix_length: int = 0,
    ) -> tuple[list[Any], list[Any], Any, list[int]]:
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
        states = [initial.detach().clone()]
        flows: list[Any] = []
        active = initial
        for step_index, timestep in enumerate(timesteps):
            flow = _call_flow_model(
                pipeline=pipeline,
                active=active,
                cond_latents=cond_latents,
                start_frame=start_frame,
                timestep=timestep,
                condition=condition,
                caches=caches,
                selected_count=len(selected),
                task_type=task_type,
                branch=branch,
                step_index=step_index,
                events=generator_events,
            )
            if prefix_length:
                if prefix_reference_flows is None:
                    raise ValueError("Prefix clamping requires reference flows")
                flow = torch.cat(
                    [
                        prefix_reference_flows[step_index][:, :, :prefix_length],
                        flow[:, :, prefix_length:],
                    ],
                    dim=2,
                )
            flows.append(flow.detach().clone())
            active = _euler_step(active, flow, sigmas, step_index)
            if prefix_length and step_index < denoising_steps - 1:
                if prefix_reference_states is None:
                    raise ValueError("Prefix clamping requires reference states")
                active = torch.cat(
                    [
                        prefix_reference_states[step_index + 1][:, :, :prefix_length],
                        active[:, :, prefix_length:],
                    ],
                    dim=2,
                )
            if step_index < denoising_steps - 1:
                states.append(active.detach().clone())
        return states, flows, active, selected

    with torch.inference_mode():
        for start_frame in range(0, frame_count, chunk_size):
            initial = latents[:, :, start_frame : start_frame + chunk_size]
            if start_frame not in events:
                _, _, student_final, _ = run_branch(
                    initial=initial,
                    start_frame=start_frame,
                    condition=requested_condition,
                    branch="uninterrupted",
                )
            else:
                event_ordinal = len(captures)
                stale_condition = stale_conditions_by_event[start_frame]
                old_states, old_flows, old_final, old_memory = run_branch(
                    initial=initial,
                    start_frame=start_frame,
                    condition=stale_condition,
                    branch="old_teacher",
                )
                intra_chunk_offset = offsets[start_frame]
                new_states, new_flows, new_final, new_memory = run_branch(
                    initial=initial,
                    start_frame=start_frame,
                    condition=requested_condition,
                    branch="rollback_teacher",
                    prefix_reference_states=old_states,
                    prefix_reference_flows=old_flows,
                    prefix_length=intra_chunk_offset,
                )
                receipt = int(runtime_receipt_steps[event_ordinal % len(runtime_receipt_steps)])
                history_native = (
                    output[:, :, start_frame - chunk_size : start_frame].detach().clone()
                )
                rollout_age = (
                    0
                    if model_config is None
                    else min(
                        event_ordinal,
                        int(getattr(model_config, "max_rollout_age", 0)),
                    )
                )
                if transport_model is None:
                    # Bootstrap captures use exact rollback history.  Train the
                    # first state corrector from these captures, then rerun this
                    # same entry point with that checkpoint for on-policy data.
                    student_final = new_final
                else:
                    active_old = spec.to_canonical(old_states[receipt])
                    zero_transition_noises = [
                        torch.zeros_like(active_old.flatten(0, 1))
                        for _ in range(denoising_steps - 1)
                    ]
                    from ..core.runtime import apply_transport_model

                    suffix_mask = None
                    if intra_chunk_offset:
                        suffix_mask = torch.zeros(
                            [batch_size, chunk_size, 1, 1, 1],
                            device=active_old.device,
                            dtype=active_old.dtype,
                        )
                        suffix_mask[:, intra_chunk_offset:] = 1.0
                    corrected, correction = apply_transport_model(
                        model=transport_model,
                        active_state=active_old,
                        initial_state=spec.to_canonical(old_states[0]),
                        history_tail=spec.to_canonical(history_native),
                        transition_noises=zero_transition_noises,
                        old_viewmats=stale_condition.viewmats[
                            :, start_frame : start_frame + chunk_size
                        ],
                        new_viewmats=requested_condition.viewmats[
                            :, start_frame : start_frame + chunk_size
                        ],
                        intrinsics=requested_condition.intrinsics[
                            :, start_frame : start_frame + chunk_size
                        ],
                        receipt_step=receipt,
                        jump_horizon=0,
                        cached_prediction=active_old,
                        rollout_age=rollout_age,
                        temporal_suffix_mask=suffix_mask,
                    )
                    correction.update(
                        event_ordinal=event_ordinal,
                        start_frame=start_frame,
                    )
                    correction_events.append(correction)

                    requested_caches, selected = _build_context_caches(
                        pipeline=pipeline,
                        output=output,
                        cond_latents=cond_latents,
                        start_frame=start_frame,
                        condition=requested_condition,
                        task_type=task_type,
                        timesteps=timesteps,
                        positive_text_cache=positive_cache,
                        negative_text_cache=negative_cache,
                        history_selector=history_selector,
                    )
                    active = spec.from_canonical(corrected)
                    if intra_chunk_offset:
                        active = torch.cat(
                            [
                                old_states[receipt][:, :, :intra_chunk_offset],
                                active[:, :, intra_chunk_offset:],
                            ],
                            dim=2,
                        )
                    for step_index in range(receipt, denoising_steps):
                        flow = _call_flow_model(
                            pipeline=pipeline,
                            active=active,
                            cond_latents=cond_latents,
                            start_frame=start_frame,
                            timestep=timesteps[step_index],
                            condition=requested_condition,
                            caches=requested_caches,
                            selected_count=len(selected),
                            task_type=task_type,
                            branch="student_cleanup",
                            step_index=step_index,
                            events=generator_events,
                        )
                        if intra_chunk_offset:
                            flow = torch.cat(
                                [
                                    old_flows[step_index][:, :, :intra_chunk_offset],
                                    flow[:, :, intra_chunk_offset:],
                                ],
                                dim=2,
                            )
                        active = _euler_step(active, flow, sigmas, step_index)
                        if intra_chunk_offset:
                            prefix_state = (
                                old_states[step_index + 1]
                                if step_index < denoising_steps - 1
                                else old_final
                            )
                            active = torch.cat(
                                [
                                    prefix_state[:, :, :intra_chunk_offset],
                                    active[:, :, intra_chunk_offset:],
                                ],
                                dim=2,
                            )
                    student_final = active

                transition_trace = torch.zeros(
                    [
                        denoising_steps - 1,
                        batch_size,
                        chunk_size,
                        channels,
                        int(initial.shape[-2]),
                        int(initial.shape[-1]),
                    ],
                    device=initial.device,
                    dtype=initial.dtype,
                )
                capture = TransportCapture(
                    tensors={
                        "old_states": spec.trajectory_to_canonical(torch.stack(old_states)),
                        "new_states": spec.trajectory_to_canonical(torch.stack(new_states)),
                        "old_predictions": spec.trajectory_to_canonical(torch.stack(old_flows)),
                        "new_predictions": spec.trajectory_to_canonical(torch.stack(new_flows)),
                        "transition_noises": transition_trace,
                        "history_tail": spec.to_canonical(history_native),
                        "old_viewmats": stale_condition.viewmats[
                            :, start_frame : start_frame + chunk_size
                        ]
                        .detach()
                        .clone(),
                        "new_viewmats": requested_condition.viewmats[
                            :, start_frame : start_frame + chunk_size
                        ]
                        .detach()
                        .clone(),
                        "intrinsics": requested_condition.intrinsics[
                            :, start_frame : start_frame + chunk_size
                        ]
                        .detach()
                        .clone(),
                        "old_actions": stale_condition.actions[
                            :, start_frame : start_frame + chunk_size
                        ]
                        .detach()
                        .clone(),
                        "new_actions": requested_condition.actions[
                            :, start_frame : start_frame + chunk_size
                        ]
                        .detach()
                        .clone(),
                        "timesteps": timesteps.detach().clone(),
                    },
                    metadata={
                        "schema_version": 1,
                        "backbone": "official_hyworld15_action2v",
                        "native_state_layout": "BCTHW",
                        "canonical_state_layout": "BTCHW",
                        "stochastic_transitions": False,
                        "transport_target_parameterization": "state",
                        "prediction_semantics": "flow_velocity",
                        "denoising_steps": denoising_steps,
                        "chunk_size": chunk_size,
                        "event_pose_index": start_frame,
                        "effective_pose_index": start_frame + intra_chunk_offset,
                        "intra_chunk_offset": intra_chunk_offset,
                        "teacher_prefix_clamped": bool(intra_chunk_offset),
                        "receipt_steps": list(range(1, denoising_steps)),
                        "teacher_history_source": "student_committed_history",
                        "recurrent_prior_corrections": event_ordinal,
                        "recurrent_event_index": event_ordinal,
                        "recurrent_rollout_age": rollout_age,
                        "runtime_receipt_step": receipt,
                        "runtime_jump_horizon": 0,
                        "cache_policy": "reconstituted_context_read_only_during_denoising",
                        "student_policy": (
                            "rollback_teacher"
                            if transport_model is None
                            else "cst_t"
                            if intra_chunk_offset
                            else "cst_r"
                        ),
                        "old_memory_indices": old_memory,
                        "new_memory_indices": new_memory,
                    },
                )
                if not intra_chunk_offset:
                    for key in (
                        "effective_pose_index",
                        "intra_chunk_offset",
                        "teacher_prefix_clamped",
                    ):
                        capture.metadata.pop(key, None)
                validate_transport_capture(capture)
                captures.append(capture)

            output[:, :, start_frame : start_frame + chunk_size] = student_final.to(output.dtype)

    return HYWorldPlayRecurrentResult(
        captures=captures,
        output=output,
        metrics={
            "correction_count": len(correction_events),
            "generator_call_count": len(generator_events),
            "generator_elapsed_ms": sum(float(event["elapsed_ms"]) for event in generator_events),
            "correction_elapsed_ms": sum(float(event["elapsed_ms"]) for event in correction_events),
            "generator_events": generator_events,
            "correction_events": correction_events,
        },
    )


class OfficialHYWorldPlayCSTHook(AbstractContextManager["OfficialHYWorldPlayCSTHook"]):
    """Replace the official pipeline's AR loop without editing upstream code."""

    def __init__(
        self,
        *,
        pipeline: Any,
        stale_conditions_by_event: dict[int, HYWorldPlayCondition],
        event_pose_indices: Sequence[int],
        runtime_receipt_steps: Sequence[int],
        transport_model: Any | None,
        intra_chunk_offsets_by_event: dict[int, int] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.stale_conditions_by_event = stale_conditions_by_event
        self.event_pose_indices = list(event_pose_indices)
        self.runtime_receipt_steps = list(runtime_receipt_steps)
        self.transport_model = transport_model
        self.intra_chunk_offsets_by_event = dict(intra_chunk_offsets_by_event or {})
        self.original_ar_rollout: Any | None = None
        self.result: HYWorldPlayRecurrentResult | None = None

    def __enter__(self) -> "OfficialHYWorldPlayCSTHook":
        self.original_ar_rollout = self.pipeline.ar_rollout

        def wrapped_ar_rollout(**kwargs: Any) -> Any:
            requested = HYWorldPlayCondition(
                viewmats=kwargs["viewmats"],
                intrinsics=kwargs["Ks"],
                actions=kwargs["action"],
            )
            enabled = bool(getattr(self.pipeline, "enable_offloading", False))
            context = nullcontext()
            if enabled:
                from hyvideo.commons import auto_offload_model

                context = auto_offload_model(
                    self.pipeline.transformer,
                    self.pipeline.execution_device,
                    enabled=True,
                )
            with context:
                self.result = capture_hyworld15_recurrent_sequence(
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
                    runtime_receipt_steps=self.runtime_receipt_steps,
                    transport_model=self.transport_model,
                    intra_chunk_offsets_by_event=(self.intra_chunk_offsets_by_event),
                )
            return self.result.output

        self.pipeline.ar_rollout = wrapped_ar_rollout
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.original_ar_rollout is not None:
            self.pipeline.ar_rollout = self.original_ar_rollout
        return None


def _initialize_text_caches(
    *,
    pipeline: Any,
    prompt_embeds: Any,
    prompt_mask: Any,
    vision_states: Any,
    task_type: str,
    extra_kwargs: dict[str, Any],
    device: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    import torch

    pipeline.init_kv_cache()
    positive_index = 1 if pipeline.do_classifier_free_guidance else 0

    def build(index: int, cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text_extra = {
            "byt5_text_states": extra_kwargs["byt5_text_states"][index, None, ...],
            "byt5_text_mask": extra_kwargs["byt5_text_mask"][index, None, ...],
        }
        timestep = torch.tensor([0], device=device, dtype=prompt_embeds.dtype)
        with _autocast_context(pipeline, device):
            return pipeline.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=timestep,
                text_states=prompt_embeds[index, None, ...],
                encoder_attention_mask=prompt_mask[index, None, ...],
                vision_states=vision_states[index, None, ...],
                mask_type=task_type,
                extra_kwargs=text_extra,
                kv_cache=cache,
                cache_txt=True,
            )

    positive = build(positive_index, pipeline._kv_cache)
    negative = build(0, pipeline._kv_cache_neg) if pipeline.do_classifier_free_guidance else None
    return positive, negative


def _build_context_caches(
    *,
    pipeline: Any,
    output: Any,
    cond_latents: Any,
    start_frame: int,
    condition: HYWorldPlayCondition,
    task_type: str,
    timesteps: Any,
    positive_text_cache: list[dict[str, Any]],
    negative_text_cache: list[dict[str, Any]] | None,
    history_selector: Callable[..., Sequence[int]],
) -> tuple[tuple[list[dict[str, Any]], list[dict[str, Any]] | None], list[int]]:
    import torch

    if start_frame == 0:
        return (
            _copy_cache(positive_text_cache),
            _copy_optional_cache(negative_text_cache),
        ), []
    selected = list(
        history_selector(
            condition.viewmats[0].detach().cpu().numpy(),
            start_frame,
            memory_frames=20,
            temporal_context_size=12,
            pred_latent_size=4,
            points_local=pipeline.points_local,
            device=output.device,
        )
    )
    selected = sorted({int(value) for value in selected if int(value) < start_frame})
    if not selected:
        raise RuntimeError("HY-WorldPlay selected no committed context frames")
    context = torch.cat(
        [output[:, :, selected], cond_latents[:, :, selected]],
        dim=1,
    )
    timestep = torch.full(
        (len(selected),),
        14,
        device=output.device,
        dtype=timesteps.dtype,
    )

    def build(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with _autocast_context(pipeline, output.device):
            return pipeline.transformer(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                hidden_states=context,
                timestep=timestep,
                timestep_r=None,
                mask_type=task_type,
                return_dict=False,
                viewmats=condition.viewmats[:, selected].to(pipeline.target_dtype),
                Ks=condition.intrinsics[:, selected].to(pipeline.target_dtype),
                action=condition.actions[:, selected].to(pipeline.target_dtype),
                kv_cache=_copy_cache(cache),
                cache_vision=True,
                rope_temporal_size=len(selected),
                start_rope_start_idx=0,
            )

    return (
        build(positive_text_cache),
        build(negative_text_cache) if negative_text_cache is not None else None,
    ), selected


def _call_flow_model(
    *,
    pipeline: Any,
    active: Any,
    cond_latents: Any,
    start_frame: int,
    timestep: Any,
    condition: HYWorldPlayCondition,
    caches: tuple[list[dict[str, Any]], list[dict[str, Any]] | None],
    selected_count: int,
    task_type: str,
    branch: str,
    step_index: int,
    events: list[dict[str, Any]],
) -> Any:
    import torch

    frame_count = int(active.shape[2])
    timestep_input = torch.full(
        (frame_count,),
        timestep,
        device=active.device,
        dtype=timestep.dtype,
    )
    hidden = torch.cat(
        [
            active,
            cond_latents[:, :, start_frame : start_frame + frame_count],
        ],
        dim=1,
    )
    viewmats = condition.viewmats[:, start_frame : start_frame + frame_count]
    intrinsics = condition.intrinsics[:, start_frame : start_frame + frame_count]
    actions = condition.actions[:, start_frame : start_frame + frame_count]

    def call(cache: list[dict[str, Any]]) -> Any:
        return pipeline.transformer(
            bi_inference=False,
            ar_txt_inference=False,
            ar_vision_inference=True,
            hidden_states=hidden,
            timestep=timestep_input,
            timestep_r=None,
            mask_type=task_type,
            return_dict=False,
            viewmats=viewmats.to(pipeline.target_dtype),
            Ks=intrinsics.to(pipeline.target_dtype),
            action=actions.to(pipeline.target_dtype),
            kv_cache=cache,
            cache_vision=False,
            rope_temporal_size=frame_count + selected_count,
            start_rope_start_idx=selected_count,
        )[0]

    if active.is_cuda:
        torch.cuda.synchronize(active.device)
    started = time.perf_counter()
    with _autocast_context(pipeline, active.device):
        positive = call(caches[0])
        if pipeline.do_classifier_free_guidance:
            if caches[1] is None:
                raise RuntimeError("Missing HY classifier-free guidance cache")
            negative = call(caches[1])
            flow = negative + pipeline.guidance_scale * (positive - negative)
        else:
            flow = positive
    if active.is_cuda:
        torch.cuda.synchronize(active.device)
    events.append(
        {
            "branch": branch,
            "start_frame": start_frame,
            "denoise_step_index": step_index,
            "timestep": float(timestep.detach().cpu().item()),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return flow


def _scheduler_sigmas(scheduler: Any, timesteps: Any) -> Any:
    import torch

    sigmas = scheduler.sigmas
    if int(sigmas.numel()) != int(timesteps.numel()) + 1:
        raise ValueError("HY scheduler sigma/timestep geometry is inconsistent")
    return sigmas.to(device=timesteps.device, dtype=torch.float32)


def _euler_step(active: Any, flow: Any, sigmas: Any, step_index: int) -> Any:
    dt = sigmas[step_index + 1] - sigmas[step_index]
    return (active.float() + flow.float() * dt).to(active.dtype)


def _copy_cache(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(layer) for layer in cache]


def _condition_to_device(
    condition: HYWorldPlayCondition,
    device: Any,
) -> HYWorldPlayCondition:
    return HYWorldPlayCondition(
        viewmats=condition.viewmats.to(device),
        intrinsics=condition.intrinsics.to(device),
        actions=condition.actions.to(device),
    )


def _copy_optional_cache(
    cache: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    return None if cache is None else _copy_cache(cache)


def _autocast_context(pipeline: Any, device: Any) -> Any:
    import torch

    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=pipeline.target_dtype,
        enabled=bool(getattr(pipeline, "autocast_enabled", True)),
    )


__all__ = [
    "HYWorldPlayCondition",
    "HYWorldPlayRecurrentResult",
    "OfficialHYWorldPlayCSTHook",
    "capture_hyworld15_recurrent_sequence",
]
