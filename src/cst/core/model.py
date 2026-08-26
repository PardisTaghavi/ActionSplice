"""Lightweight residual model for counterfactual active-state transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransportModelConfig:
    latent_channels: int = 16
    denoising_steps: int = 4
    base_channels: int = 64
    condition_channels: int = 256
    condition_frame_features: int = 45
    max_jump_horizon: int = 0
    use_cached_prediction: bool = False
    target_parameterization: str = "state"
    iterated_one_step: bool = False
    endpoint_noise_exclusive: bool = False
    relative_pose_target_from_source: bool = False
    max_rollout_age: int = 0
    transport_role: str = "generic"
    use_temporal_suffix_mask: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CameraConditionEncoder(nn.Module):
    """Encode source, target, relative camera, and intrinsics per frame."""

    def __init__(
        self,
        feature_channels: int,
        condition_channels: int,
        *,
        relative_pose_target_from_source: bool,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.relative_pose_target_from_source = relative_pose_target_from_source
        self.frame_mlp = nn.Sequential(
            nn.Linear(feature_channels, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(condition_channels * 3, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
        )

    def forward(
        self,
        old_viewmats: torch.Tensor,
        new_viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        features = camera_frame_features(
            old_viewmats,
            new_viewmats,
            intrinsics,
            target_from_source=self.relative_pose_target_from_source,
        )
        if int(features.shape[-1]) != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} camera features, received {features.shape[-1]}"
            )
        frame_embeddings = self.frame_mlp(features)
        pooled = torch.cat(
            [
                frame_embeddings.mean(dim=1),
                frame_embeddings[:, 0],
                frame_embeddings[:, -1],
            ],
            dim=-1,
        )
        return self.pool_projection(pooled)


def camera_frame_features(
    old_viewmats: torch.Tensor,
    new_viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    target_from_source: bool = True,
) -> torch.Tensor:
    """Return 45 model-independent camera features per latent frame."""
    if old_viewmats.shape != new_viewmats.shape:
        raise ValueError("Source and target viewmat shapes differ")
    if old_viewmats.ndim != 4 or old_viewmats.shape[-2:] != (4, 4):
        raise ValueError("Viewmats must have shape [B, T, 4, 4]")
    if intrinsics.shape[:2] != old_viewmats.shape[:2]:
        raise ValueError("Intrinsics and viewmats must align in batch and time")
    old = old_viewmats.float()
    new = new_viewmats.float()
    relative = new @ torch.linalg.inv(old) if target_from_source else torch.linalg.solve(old, new)
    old_flat = old[..., :3, :4].flatten(start_dim=-2)
    new_flat = new[..., :3, :4].flatten(start_dim=-2)
    relative_flat = relative[..., :3, :4].flatten(start_dim=-2)
    intrinsics_flat = intrinsics.float().flatten(start_dim=-2)
    intrinsic_scale = intrinsics_flat.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    intrinsics_normalized = intrinsics_flat / intrinsic_scale
    return torch.cat(
        [old_flat, new_flat, relative_flat, intrinsics_normalized],
        dim=-1,
    )


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.spatial = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.film = nn.Linear(condition_channels, channels * 2)
        self.temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.spatial(F.silu(self.norm1(x)))
        hidden = self.norm2(hidden)
        scale, shift = self.film(condition).chunk(2, dim=-1)
        scale = scale.to(hidden.dtype)[..., None, None, None]
        shift = shift.to(hidden.dtype)[..., None, None, None]
        hidden = hidden * (1.0 + scale) + shift
        hidden = self.temporal(F.silu(hidden))
        return x + hidden


class CounterfactualTransport(nn.Module):
    """Transport z_r^- toward the matched new-condition state z_r^+."""

    def __init__(self, config: TransportModelConfig) -> None:
        super().__init__()
        if config.denoising_steps < 2:
            raise ValueError("denoising_steps must be at least two")
        if config.max_jump_horizon != 0:
            raise ValueError("CST-R/CST-T operate at the current solver step")
        if config.target_parameterization not in {"state", "clean_prediction"}:
            raise ValueError("target_parameterization must be state or clean_prediction")
        if (
            config.target_parameterization == "clean_prediction"
            and not config.use_cached_prediction
        ):
            raise ValueError("clean_prediction requires use_cached_prediction=True")
        if config.max_rollout_age < 0:
            raise ValueError("max_rollout_age must be nonnegative")
        if config.transport_role not in {
            "generic",
            "action_h0",
            "action_hm",
            "action_h0_state",
            "action_hm_state",
        }:
            raise ValueError("Unknown transport_role")
        if config.transport_role == "action_h0" and (
            config.max_jump_horizon != 0 or config.target_parameterization != "clean_prediction"
        ):
            raise ValueError("action_h0 must be a clean-prediction checkpoint with horizon 0")
        if config.transport_role == "action_hm" and (
            config.max_jump_horizon != 0
            or config.target_parameterization != "clean_prediction"
            or not config.use_temporal_suffix_mask
        ):
            raise ValueError(
                "action_hm must be a masked clean-prediction checkpoint with horizon 0"
            )
        if config.transport_role == "action_h0_state" and (
            config.max_jump_horizon != 0 or config.target_parameterization != "state"
        ):
            raise ValueError("action_h0_state must be a direct-state checkpoint with horizon 0")
        if config.transport_role == "action_hm_state" and (
            config.max_jump_horizon != 0
            or config.target_parameterization != "state"
            or not config.use_temporal_suffix_mask
        ):
            raise ValueError(
                "action_hm_state must be a masked direct-state checkpoint with horizon 0"
            )
        self.config = config
        latent_inputs = 3 + config.denoising_steps - 1 + int(config.use_cached_prediction)
        input_channels = config.latent_channels * latent_inputs + int(
            config.use_temporal_suffix_mask
        )
        base = config.base_channels
        condition = config.condition_channels

        self.camera_encoder = CameraConditionEncoder(
            config.condition_frame_features,
            condition,
            relative_pose_target_from_source=(config.relative_pose_target_from_source),
        )
        self.receipt_embedding = nn.Embedding(
            config.denoising_steps,
            condition,
        )
        # Kept as an attribute so zero-horizon checkpoints retain their exact
        # state-dict structure. CST-R/CST-T never create horizon weights.
        self.horizon_embedding = None
        self.rollout_age_embedding = (
            nn.Embedding(config.max_rollout_age + 1, condition)
            if config.max_rollout_age > 0
            else None
        )
        self.stem = nn.Conv3d(input_channels, base, kernel_size=1)
        self.level0 = FiLMResidualBlock(base, condition)
        self.down1 = nn.Conv3d(
            base,
            base * 2,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.level1 = FiLMResidualBlock(base * 2, condition)
        self.down2 = nn.Conv3d(
            base * 2,
            base * 4,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.mid1 = FiLMResidualBlock(base * 4, condition)
        self.mid2 = FiLMResidualBlock(base * 4, condition)
        self.up1_projection = nn.Conv3d(base * 4, base * 2, kernel_size=1)
        self.up1 = FiLMResidualBlock(base * 2, condition)
        self.up0_projection = nn.Conv3d(base * 2, base, kernel_size=1)
        self.up0 = FiLMResidualBlock(base, condition)
        self.output_norm = nn.GroupNorm(min(32, base), base)
        self.output = nn.Conv3d(
            base,
            config.latent_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        *,
        active_state: torch.Tensor,
        initial_state: torch.Tensor,
        history_tail: torch.Tensor,
        noise_trace: torch.Tensor,
        old_viewmats: torch.Tensor,
        new_viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        receipt_step: torch.Tensor,
        jump_horizon: torch.Tensor | None = None,
        cached_prediction: torch.Tensor | None = None,
        rollout_age: torch.Tensor | None = None,
        temporal_suffix_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        jump_horizon = self._normalize_jump_horizon(
            receipt_step,
            jump_horizon,
        )
        rollout_age = self._normalize_rollout_age(receipt_step, rollout_age)
        self._validate_latents(
            active_state,
            initial_state,
            history_tail,
            noise_trace,
            receipt_step,
            jump_horizon,
            cached_prediction,
            rollout_age,
            temporal_suffix_mask,
        )
        condition = self.camera_encoder(
            old_viewmats,
            new_viewmats,
            intrinsics,
        )
        condition = condition + self.receipt_embedding(receipt_step)
        if self.horizon_embedding is not None:
            condition = condition + self.horizon_embedding(jump_horizon)
        if self.rollout_age_embedding is not None:
            condition = condition + self.rollout_age_embedding(rollout_age)
        model_input = self._build_model_input(
            active_state,
            initial_state,
            history_tail,
            noise_trace,
            receipt_step,
            jump_horizon,
            cached_prediction,
            temporal_suffix_mask,
        )

        level0 = self.level0(self.stem(model_input), condition)
        level1 = self.level1(self.down1(level0), condition)
        middle = self.mid2(self.mid1(self.down2(level1), condition), condition)
        up1 = F.interpolate(
            middle,
            size=level1.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        up1 = self.up1(self.up1_projection(up1) + level1, condition)
        up0 = F.interpolate(
            up1,
            size=level0.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        up0 = self.up0(self.up0_projection(up0) + level0, condition)
        output_features = F.silu(self.output_norm(up0))
        residual_base = (
            cached_prediction
            if self.config.target_parameterization == "clean_prediction"
            else active_state
        )
        if residual_base is None:
            raise RuntimeError("Missing residual base")
        delta_channels_first = self.output(output_features)
        delta = delta_channels_first.permute(0, 2, 1, 3, 4).contiguous()
        if self.config.use_temporal_suffix_mask:
            if temporal_suffix_mask is None:
                raise RuntimeError("Masked transport requires a temporal suffix mask")
            delta = delta * temporal_suffix_mask.to(delta.dtype)
        predicted_target = residual_base + delta
        return {
            "corrected_state": predicted_target,
            "predicted_target": predicted_target,
            "delta": delta,
        }

    def _build_model_input(
        self,
        active_state: torch.Tensor,
        initial_state: torch.Tensor,
        history_tail: torch.Tensor,
        noise_trace: torch.Tensor,
        receipt_step: torch.Tensor,
        jump_horizon: torch.Tensor,
        cached_prediction: torch.Tensor | None = None,
        temporal_suffix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, trace_steps, frames, channels, height, width = noise_trace.shape
        step_indices = torch.arange(
            trace_steps,
            device=receipt_step.device,
        )[None, :]
        target_step = receipt_step + jump_horizon
        visible_transition_count = (
            target_step - 1 if self.config.endpoint_noise_exclusive else target_step
        )
        mask = (step_indices < visible_transition_count[:, None]).to(noise_trace.dtype)
        masked_trace = noise_trace * mask[:, :, None, None, None, None]
        trace_channels = masked_trace.permute(0, 2, 1, 3, 4, 5).reshape(
            batch,
            frames,
            trace_steps * channels,
            height,
            width,
        )
        latent_inputs = [
            active_state,
            initial_state,
            history_tail,
            trace_channels,
        ]
        if self.config.use_cached_prediction:
            if cached_prediction is None:
                raise ValueError("cached_prediction is required by this checkpoint")
            latent_inputs.append(cached_prediction)
        combined = torch.cat(latent_inputs, dim=2)
        if self.config.use_temporal_suffix_mask:
            if temporal_suffix_mask is None:
                raise ValueError("temporal_suffix_mask is required by this checkpoint")
            spatial_mask = temporal_suffix_mask.to(combined.dtype).expand(
                -1,
                -1,
                -1,
                int(combined.shape[-2]),
                int(combined.shape[-1]),
            )
            combined = torch.cat([combined, spatial_mask], dim=2)
        return combined.permute(0, 2, 1, 3, 4).contiguous()

    def _validate_latents(
        self,
        active_state: torch.Tensor,
        initial_state: torch.Tensor,
        history_tail: torch.Tensor,
        noise_trace: torch.Tensor,
        receipt_step: torch.Tensor,
        jump_horizon: torch.Tensor,
        cached_prediction: torch.Tensor | None,
        rollout_age: torch.Tensor,
        temporal_suffix_mask: torch.Tensor | None,
    ) -> None:
        if active_state.ndim != 5:
            raise ValueError("Latent states must have shape [B, T, C, H, W]")
        if active_state.shape != initial_state.shape:
            raise ValueError("active_state and initial_state shapes differ")
        if active_state.shape != history_tail.shape:
            raise ValueError("history_tail and active_state shapes differ")
        if cached_prediction is not None and (cached_prediction.shape != active_state.shape):
            raise ValueError("cached_prediction and active_state shapes differ")
        if int(active_state.shape[2]) != self.config.latent_channels:
            raise ValueError("Latent channel count does not match configuration")
        expected_trace = self.config.denoising_steps - 1
        if noise_trace.ndim != 6 or int(noise_trace.shape[1]) != expected_trace:
            raise ValueError("noise_trace has the wrong denoising-step dimension")
        if noise_trace.shape[0] != active_state.shape[0]:
            raise ValueError("noise_trace batch does not match active_state")
        if noise_trace.shape[2:] != active_state.shape[1:]:
            raise ValueError("noise_trace latent geometry does not match")
        if receipt_step.shape != (active_state.shape[0],):
            raise ValueError("receipt_step must have shape [B]")
        if jump_horizon.shape != receipt_step.shape:
            raise ValueError("jump_horizon must match receipt_step")
        if rollout_age.shape != receipt_step.shape:
            raise ValueError("rollout_age must match receipt_step")
        if self.config.use_temporal_suffix_mask:
            expected_mask_shape = (
                int(active_state.shape[0]),
                int(active_state.shape[1]),
                1,
                1,
                1,
            )
            if (
                temporal_suffix_mask is None
                or tuple(temporal_suffix_mask.shape) != expected_mask_shape
            ):
                raise ValueError("temporal_suffix_mask must have shape [B, T, 1, 1, 1]")
            transitions = temporal_suffix_mask[:, 1:] - temporal_suffix_mask[:, :-1]
            if bool((temporal_suffix_mask < 0).logical_or(temporal_suffix_mask > 1).any()):
                raise ValueError("temporal_suffix_mask values must be in [0, 1]")
            if bool((transitions < 0).any()):
                raise ValueError("temporal_suffix_mask must select a temporal suffix")
        if bool(((receipt_step <= 0) | (receipt_step >= self.config.denoising_steps)).any()):
            raise ValueError("receipt_step is outside the legal range")
        if bool(
            (
                (jump_horizon < 0)
                | (jump_horizon > self.config.max_jump_horizon)
                | (receipt_step + jump_horizon > self.config.denoising_steps)
            ).any()
        ):
            raise ValueError("jump_horizon is outside the legal range")
        if bool((rollout_age < 0).logical_or(rollout_age > self.config.max_rollout_age).any()):
            raise ValueError("rollout_age is outside the configured range")

    def _normalize_jump_horizon(
        self,
        receipt_step: torch.Tensor,
        jump_horizon: torch.Tensor | None,
    ) -> torch.Tensor:
        if jump_horizon is None:
            return torch.zeros_like(receipt_step)
        return jump_horizon.to(
            device=receipt_step.device,
            dtype=torch.long,
        )

    def _normalize_rollout_age(
        self,
        receipt_step: torch.Tensor,
        rollout_age: torch.Tensor | None,
    ) -> torch.Tensor:
        if rollout_age is None:
            return torch.zeros_like(receipt_step)
        return rollout_age.to(
            device=receipt_step.device,
            dtype=torch.long,
        )


def normalized_state_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    dimensions = tuple(range(1, prediction.ndim))
    numerator = (prediction.float() - target.float()).square().mean(dim=dimensions)
    denominator = target.float().square().mean(dim=dimensions).clamp_min(epsilon)
    return (numerator / denominator).mean()


def reconstruct_noisy_state(
    clean_prediction: torch.Tensor,
    transition_noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact scalar flow-matching noise transition."""
    if clean_prediction.shape != transition_noise.shape:
        raise ValueError("Clean prediction and transition noise shapes differ")
    if sigma.ndim != 1 or sigma.shape[0] != clean_prediction.shape[0]:
        raise ValueError("sigma must have shape [B]")
    expanded_sigma = sigma.to(
        device=clean_prediction.device,
        dtype=clean_prediction.dtype,
    ).reshape(-1, 1, 1, 1, 1)
    return (1.0 - expanded_sigma) * clean_prediction + expanded_sigma * transition_noise


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_payload(
    model: CounterfactualTransport,
    *,
    optimizer: Any | None,
    step: int,
    validation_nmse: float,
) -> dict[str, Any]:
    return {
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": int(step),
        "validation_nmse": float(validation_nmse),
    }
