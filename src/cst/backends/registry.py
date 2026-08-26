"""Registry for the two verified CST backends."""

from __future__ import annotations

from typing import Any, Mapping, cast

from ..core.state import HY15_ACTION2V_STATE_SPEC, WAN_ACTION2V_STATE_SPEC
from .base import BackendSpec, CSTMethod

BACKENDS: dict[str, BackendSpec] = {
    "minwm": BackendSpec(
        name="minwm",
        display_name="minWM (Wan Action2V)",
        state_spec=WAN_ACTION2V_STATE_SPEC,
        target_parameterization="clean_prediction",
        checkpoint_roles={
            "cst_r": "action_h0",
            "cst_t": "action_hm",
        },
    ),
    "hyworld15": BackendSpec(
        name="hyworld15",
        display_name="HY-WM1.5 / HY-WorldPlay",
        state_spec=HY15_ACTION2V_STATE_SPEC,
        target_parameterization="state",
        checkpoint_roles={
            "cst_r": "action_h0_state",
            "cst_t": "action_hm_state",
        },
    ),
}


def get_backend(name: str) -> BackendSpec:
    """Return one backend without importing either upstream repository."""
    normalized = name.strip().lower().replace("-", "")
    aliases = {
        "minwm": "minwm",
        "wan": "minwm",
        "hy": "hyworld15",
        "hywm15": "hyworld15",
        "hyworld15": "hyworld15",
    }
    try:
        return BACKENDS[aliases[normalized]]
    except KeyError as error:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}") from error


def validate_training_config(config: Mapping[str, Any]) -> tuple[BackendSpec, CSTMethod]:
    """Validate public names against checkpoint-compatible internal fields."""
    obsolete = {
        "composition_action_checkpoint",
        "composition_temporal_checkpoint",
        "initialize_expand_horizon",
        "iterated_one_step",
        "rollout_intermediate_loss_weight",
        "rollout_on_policy_teacher_checkpoint",
        "rollout_on_policy_teacher_loss_weight",
        "rollout_teacher_forcing_start_probability",
        "rollout_teacher_forcing_end_probability",
    }.intersection(config)
    if obsolete:
        raise ValueError(f"Obsolete multi-horizon training options: {sorted(obsolete)}")
    if "backend" not in config or "method" not in config:
        raise ValueError("Public training configs require 'backend' and 'method'")
    backend = get_backend(str(config["backend"]))
    method_value = str(config["method"]).lower()
    if method_value not in {"cst_r", "cst_t"}:
        raise ValueError("method must be 'cst_r' or 'cst_t'")
    method = cast(CSTMethod, method_value)
    backend.validate_checkpoint_config(config, method=method)
    horizons = tuple(int(value) for value in config.get("jump_horizons", [0]))
    if horizons != (0,):
        raise ValueError("CST-R/CST-T training requires jump_horizons=[0]")
    return backend, method


__all__ = ["BACKENDS", "get_backend", "validate_training_config"]
