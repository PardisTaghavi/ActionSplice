"""Small, dependency-free contract shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..core.state import TransportStateSpec

CSTMethod = Literal["cst_r", "cst_t"]


def _display_method(method: CSTMethod) -> str:
    return "CST-R" if method == "cst_r" else "CST-T"


@dataclass(frozen=True)
class BackendSpec:
    """Static CST properties of one upstream world-model backend.

    Upstream pipelines stay in their backend modules. This object centralizes
    only the values that must agree across training and inference.
    """

    name: str
    display_name: str
    state_spec: TransportStateSpec
    target_parameterization: Literal["clean_prediction", "state"]
    checkpoint_roles: Mapping[CSTMethod, str]

    def role(self, method: CSTMethod) -> str:
        if method not in self.checkpoint_roles:
            raise ValueError(f"Unknown CST method {method!r}")
        return self.checkpoint_roles[method]

    def method_for_role(self, role: str) -> CSTMethod:
        for method, expected_role in self.checkpoint_roles.items():
            if role == expected_role:
                return method
        raise ValueError(
            f"Checkpoint role {role!r} is not a {self.display_name} CST-R or CST-T checkpoint"
        )

    def validate_checkpoint_config(
        self,
        model_config: Mapping[str, Any],
        *,
        method: CSTMethod | None = None,
    ) -> CSTMethod:
        role = str(model_config.get("transport_role", ""))
        resolved_method = self.method_for_role(role)
        if method is not None and resolved_method != method:
            raise ValueError(
                f"Expected {_display_method(method)}, but checkpoint role {role!r} "
                f"encodes {_display_method(resolved_method)}"
            )
        parameterization = str(model_config.get("target_parameterization", ""))
        if parameterization != self.target_parameterization:
            raise ValueError(
                f"{self.display_name} requires target_parameterization="
                f"{self.target_parameterization!r}, received {parameterization!r}"
            )
        if "latent_channels" in model_config:
            channels = int(model_config["latent_channels"])
            if channels != self.state_spec.latent_channels:
                raise ValueError(
                    f"{self.display_name} requires {self.state_spec.latent_channels} "
                    f"latent channels, received {channels}"
                )
        if "denoising_steps" in model_config:
            steps = int(model_config["denoising_steps"])
            if steps != self.state_spec.denoising_steps:
                raise ValueError(
                    f"{self.display_name} requires {self.state_spec.denoising_steps} "
                    f"denoising steps, received {steps}"
                )
        if int(model_config.get("max_jump_horizon", 0)) != 0:
            raise ValueError("CST-R/CST-T checkpoints must use the current solver step")
        masked = bool(model_config.get("use_temporal_suffix_mask", False))
        if masked != (resolved_method == "cst_t"):
            raise ValueError("Only CST-T checkpoints use the hard temporal suffix mask")
        return resolved_method


__all__ = ["BackendSpec", "CSTMethod"]
