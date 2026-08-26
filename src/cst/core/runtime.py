"""Checkpoint loading and same-step CST-R/CST-T application."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def load_transport_model(
    checkpoint_path: Path,
    *,
    device: Any,
    dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load a CST-R/CST-T checkpoint without loading a backbone."""
    import torch

    from .model import CounterfactualTransport, TransportModelConfig

    checkpoint_path = checkpoint_path.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in payload or "model" not in payload:
        raise ValueError(f"Malformed transport checkpoint {checkpoint_path}")
    config = TransportModelConfig(**payload["model_config"])
    if config.transport_role not in {
        "action_h0",
        "action_hm",
        "action_h0_state",
        "action_hm_state",
    }:
        raise ValueError("Checkpoint is not a CST-R or CST-T model")
    if int(config.max_jump_horizon) != 0:
        raise ValueError("CST-R/CST-T checkpoints cannot skip solver steps")
    model = CounterfactualTransport(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "model_config": config.to_dict(),
        "training_step": int(payload.get("step", 0)),
        "validation_nmse": float(payload.get("validation_nmse", float("nan"))),
    }
    return model, metadata


def apply_transport_model(
    *,
    model: Any,
    active_state: Any,
    initial_state: Any,
    history_tail: Any,
    transition_noises: list[Any],
    old_viewmats: Any,
    new_viewmats: Any,
    intrinsics: Any,
    receipt_step: int,
    jump_horizon: int = 0,
    cached_prediction: Any | None = None,
    scheduler: Any | None = None,
    denoising_step_list: list[Any] | tuple[Any, ...] | None = None,
    rollout_age: int = 0,
    temporal_suffix_mask: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply CST-R or CST-T at step ``receipt_step``.

    ``jump_horizon`` is retained only for checkpoint/capture compatibility and
    must be zero. CST-T is selected by loading a CST-T checkpoint and
    supplying its hard temporal suffix mask.

    minWM checkpoints predict the requested-action clean prediction, which is
    re-noised at the same solver step with the stored transition noise.
    HY-WM1.5 checkpoints predict the direct Euler solver state.
    """
    import torch

    if int(jump_horizon) != 0:
        raise ValueError("CST-R/CST-T run at the current solver step; jump_horizon must be 0")
    role = str(getattr(model.config, "transport_role", ""))
    temporal_splice = role in {"action_hm", "action_hm_state"}
    if temporal_splice != (temporal_suffix_mask is not None):
        raise ValueError("CST-T requires a suffix mask; CST-R does not use one")
    if not transition_noises:
        raise ValueError("Transport requires the sampler transition-noise bank")
    batch_size = int(active_state.shape[0])
    noise_trace = _stack_noise_trace(transition_noises, active_state=active_state)
    receipt = torch.full(
        (batch_size,), int(receipt_step), device=active_state.device, dtype=torch.long
    )
    horizon = torch.zeros_like(receipt)
    age = torch.full_like(receipt, int(rollout_age))
    if active_state.is_cuda:
        torch.cuda.synchronize(active_state.device)
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=active_state.device.type,
            dtype=torch.bfloat16,
            enabled=active_state.is_cuda,
        ),
    ):
        result = model(
            active_state=active_state,
            initial_state=initial_state,
            history_tail=history_tail,
            noise_trace=noise_trace,
            old_viewmats=old_viewmats,
            new_viewmats=new_viewmats,
            intrinsics=intrinsics,
            receipt_step=receipt,
            jump_horizon=horizon,
            cached_prediction=(
                cached_prediction if cached_prediction is not None else active_state
            ),
            rollout_age=age,
            temporal_suffix_mask=temporal_suffix_mask,
        )
    predicted_target = result["predicted_target"].to(active_state.dtype)
    target_step = int(receipt_step)
    target_parameterization = str(model.config.target_parameterization)
    re_noised = False
    re_noise_timestep: int | None = None
    if target_parameterization == "clean_prediction":
        if scheduler is None or denoising_step_list is None:
            raise ValueError(
                "minWM clean-prediction transport requires the scheduler "
                "and denoising-step list for exact re-noising"
            )
        denoising_steps = len(denoising_step_list)
        if not 0 < target_step <= denoising_steps:
            raise ValueError(f"Transport step {target_step} is outside [1, {denoising_steps}]")
        if target_step < denoising_steps:
            noise_index = target_step - 1
            if noise_index >= len(transition_noises):
                raise ValueError(
                    "Transition-noise bank does not contain the target-step "
                    f"noise at index {noise_index}"
                )
            timestep_value = denoising_step_list[target_step]
            frame_count = int(predicted_target.shape[1])
            timestep = timestep_value * torch.ones(
                [batch_size * frame_count],
                device=predicted_target.device,
                dtype=torch.long,
            )
            corrected = scheduler.add_noise(
                predicted_target.flatten(0, 1),
                transition_noises[noise_index],
                timestep,
            ).unflatten(0, predicted_target.shape[:2])
            re_noised = True
            re_noise_timestep = int(timestep_value)
        else:
            corrected = predicted_target
    else:
        corrected = result["corrected_state"].to(active_state.dtype)
    if active_state.is_cuda:
        torch.cuda.synchronize(active_state.device)
    ended = time.perf_counter()
    active_float = active_state.float()
    delta_float = corrected.float() - active_float
    relative_l2 = float(
        (delta_float.square().mean().sqrt() / active_float.square().mean().sqrt().clamp_min(1e-8))
        .detach()
        .cpu()
        .item()
    )
    return corrected, {
        "method": "cst_t" if temporal_splice else "cst_r",
        "receipt_step": int(receipt_step),
        "target_step": target_step,
        "target_parameterization": target_parameterization,
        "exact_re_noised": re_noised,
        "re_noise_step_index": target_step if re_noised else None,
        "re_noise_timestep": re_noise_timestep,
        "elapsed_ms": (ended - started) * 1000.0,
        "relative_delta_l2": relative_l2,
        "rollout_age": int(rollout_age),
        "intra_chunk_offset": (
            _suffix_mask_offset(temporal_suffix_mask) if temporal_suffix_mask is not None else None
        ),
    }


def _suffix_mask_offset(mask: Any) -> int:
    values = mask[0, :, 0, 0, 0].detach().float().cpu().tolist()
    for index, value in enumerate(values):
        if value > 0.5:
            return index
    raise ValueError("Temporal suffix mask contains no editable frame")


def _stack_noise_trace(transition_noises: list[Any], *, active_state: Any) -> Any:
    import torch

    if not transition_noises:
        raise ValueError("Sampler transition-noise bank is empty")
    return torch.stack(
        [noise.unflatten(0, active_state.shape[:2]) for noise in transition_noises], dim=1
    )
