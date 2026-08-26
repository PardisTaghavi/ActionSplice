"""Backbone-specific tensor layout boundary for model-agnostic CST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

NativeLayout = Literal["BTCHW", "BCTHW"]


@dataclass(frozen=True)
class TransportStateSpec:
    backbone: str
    latent_channels: int
    denoising_steps: int
    native_layout: NativeLayout
    stochastic_transitions: bool

    def __post_init__(self) -> None:
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if self.denoising_steps < 2:
            raise ValueError("denoising_steps must be at least two")
        if self.native_layout not in {"BTCHW", "BCTHW"}:
            raise ValueError(f"Unsupported native layout {self.native_layout!r}")

    def to_canonical(self, state: Any) -> Any:
        """Convert one native state to canonical [B,T,C,H,W]."""
        if state.ndim != 5:
            raise ValueError("Native state must be rank five")
        canonical = (
            state if self.native_layout == "BTCHW" else state.permute(0, 2, 1, 3, 4).contiguous()
        )
        if int(canonical.shape[2]) != self.latent_channels:
            raise ValueError(
                f"{self.backbone} expected {self.latent_channels} channels, "
                f"received {canonical.shape[2]}"
            )
        return canonical

    def from_canonical(self, state: Any) -> Any:
        """Convert canonical [B,T,C,H,W] back to the backbone layout."""
        if state.ndim != 5:
            raise ValueError("Canonical state must be rank five")
        if int(state.shape[2]) != self.latent_channels:
            raise ValueError("Canonical state has the wrong channel count")
        return state if self.native_layout == "BTCHW" else state.permute(0, 2, 1, 3, 4).contiguous()

    def trajectory_to_canonical(self, states: Any) -> Any:
        """Convert [K,B,...] native states to [K,B,T,C,H,W]."""
        if states.ndim != 6:
            raise ValueError("State trajectory must be rank six")
        if int(states.shape[0]) != self.denoising_steps:
            raise ValueError("State trajectory has the wrong step count")
        canonical = (
            states
            if self.native_layout == "BTCHW"
            else states.permute(0, 1, 3, 2, 4, 5).contiguous()
        )
        if int(canonical.shape[3]) != self.latent_channels:
            raise ValueError("State trajectory has the wrong channel count")
        return canonical

    def transition_trace_to_canonical(
        self,
        transition_states: Sequence[Any],
        *,
        reference_state: Any,
    ) -> Any:
        """Return fixed [K-1,B,T,C,H,W] trace.

        Wan stores sampled transition noise. HY-WM1.5 Euler transitions are
        deterministic, so the same interface is represented by an all-zero
        trace rather than inventing stochastic state.
        """
        import torch

        reference = self.to_canonical(reference_state)
        expected = self.denoising_steps - 1
        if not self.stochastic_transitions:
            if transition_states:
                raise ValueError("Deterministic transport spec cannot consume noise states")
            return torch.zeros(
                [expected, *reference.shape],
                device=reference.device,
                dtype=reference.dtype,
            )
        if len(transition_states) != expected:
            raise ValueError(
                f"Expected {expected} transition states, received {len(transition_states)}"
            )
        canonical = torch.stack(
            [self.to_canonical(state) for state in transition_states],
            dim=0,
        )
        if canonical.shape[1:] != reference.shape:
            raise ValueError("Transition trace geometry differs from state")
        return canonical


WAN_ACTION2V_STATE_SPEC = TransportStateSpec(
    backbone="minwm_wan_action2v",
    latent_channels=16,
    denoising_steps=4,
    native_layout="BTCHW",
    stochastic_transitions=True,
)


HY15_ACTION2V_STATE_SPEC = TransportStateSpec(
    backbone="official_hyworld15_action2v",
    latent_channels=32,
    denoising_steps=4,
    native_layout="BCTHW",
    stochastic_transitions=False,
)
