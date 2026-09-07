"""Train the lightweight counterfactual active-state transport model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LossWeights:
    """Paper loss coefficients resolved from a training configuration."""

    lambda_state: float
    lambda_residual: float
    lambda_lpips: float
    lambda_temporal: float
    lambda_history_boundary: float
    lambda_mid: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


_LEGACY_LOSS_KEYS = {
    "lambda_residual": "delta_loss_weight",
    "lambda_lpips": "decoded_lpips_loss_weight",
    "lambda_temporal": "decoded_temporal_loss_weight",
    "lambda_history_boundary": "decoded_boundary_loss_weight",
    "lambda_mid": "intra_chunk_boundary_loss_weight",
}


def _resolve_loss_weights(config: dict[str, Any]) -> LossWeights:
    """Resolve configurable lambdas with the current HY objective as default."""

    method = str(config.get("method", "cst_r"))
    defaults = {
        "lambda_state": 1.0,
        "lambda_residual": 0.0,
        "lambda_lpips": 0.05,
        "lambda_temporal": 0.1,
        "lambda_history_boundary": 0.1,
        "lambda_mid": 0.1 if method == "cst_t" else 0.0,
    }
    configured = config.get("loss_weights", {})
    if not isinstance(configured, dict):
        raise ValueError("loss_weights must be a JSON object")
    unknown = sorted(set(configured) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown loss weights: {unknown}")

    resolved: dict[str, float] = {}
    for name, default in defaults.items():
        legacy_name = _LEGACY_LOSS_KEYS.get(name)
        if name in configured and legacy_name is not None and legacy_name in config:
            raise ValueError(
                f"Specify loss_weights.{name} or legacy {legacy_name}, not both"
            )
        value = (
            configured[name]
            if name in configured
            else config[legacy_name]
            if legacy_name is not None and legacy_name in config
            else default
        )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"loss_weights.{name} must be numeric") from error
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"loss_weights.{name} must be finite and nonnegative")
        resolved[name] = numeric

    if method == "cst_r" and resolved["lambda_mid"] != 0.0:
        raise ValueError("CST-R requires loss_weights.lambda_mid=0")
    if not any(value > 0.0 for value in resolved.values()):
        raise ValueError("At least one loss weight must be positive")
    return LossWeights(**resolved)


class _MetricLogger:
    """Append durable JSONL metrics and optionally mirror them to W&B."""

    def __init__(
        self,
        *,
        output_dir: Path,
        config: dict[str, Any],
        run_name: str,
    ) -> None:
        self.path = output_dir / "metrics.jsonl"
        self.wandb_status_path = output_dir / "wandb_status.json"
        self.wandb_run: Any | None = None
        self._wandb_log_failed = False
        project = os.environ.get("WANDB_PROJECT") or config.get("wandb_project")
        if project:
            try:
                import wandb
            except ImportError as error:
                self._write_wandb_status("disabled", error)
                warnings.warn(
                    f"W&B is unavailable; continuing with metrics.jsonl: {error}",
                    stacklevel=2,
                )
                return
            try:
                self.wandb_run = wandb.init(
                    project=str(project),
                    entity=os.environ.get("WANDB_ENTITY") or config.get("wandb_entity"),
                    name=os.environ.get("WANDB_RUN_NAME") or run_name,
                    mode=os.environ.get("WANDB_MODE", "online"),
                    dir=str(output_dir),
                    config=config,
                    resume="allow",
                    settings=wandb.Settings(
                        init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "30"))
                    ),
                )
            except Exception as error:  # W&B must never abort training.
                self._write_wandb_status("initialization_failed", error)
                warnings.warn(
                    f"W&B initialization failed; continuing with metrics.jsonl: {error}",
                    stacklevel=2,
                )
                self.wandb_run = None
            else:
                self._write_wandb_status("active")

    def _write_wandb_status(
        self,
        status: str,
        error: Exception | None = None,
    ) -> None:
        payload = {
            "status": status,
            "mode": os.environ.get("WANDB_MODE", "online"),
        }
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"
        self.wandb_status_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def log(self, record: dict[str, Any], *, step: int) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(record, step=step)
            except Exception as error:  # Preserve training on telemetry failure.
                if not self._wandb_log_failed:
                    warnings.warn(
                        f"W&B logging failed; continuing with metrics.jsonl: {error}",
                        stacklevel=2,
                    )
                    self._wandb_log_failed = True
                self._write_wandb_status("logging_failed", error)
                self.wandb_run = None

    def finish(self) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as error:  # Metrics are already durable in JSONL.
                self._write_wandb_status("finish_failed", error)
                warnings.warn(
                    f"W&B finish failed: {error}",
                    stacklevel=2,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--additional-capture-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional teacher-labelled capture directory; repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--group-by",
        help=(
            "Keep this capture metadata field intact in the two-way "
            "train/validation split, e.g. 'prompt'."
        ),
    )
    parser.add_argument("--resume", type=Path)
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "seed",
        "validation_fraction",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_steps",
        "validation_interval",
        "num_workers",
        "base_channels",
        "condition_channels",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Training config is missing: {missing}")
    return config


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    from ..backends import validate_training_config
    from ..core.model import (
        CounterfactualTransport,
        TransportModelConfig,
        checkpoint_payload,
        model_parameter_count,
    )
    from ..data.dataset import (
        CaptureSplit,
        RecurrentSequenceBatchSampler,
        TransportPairDataset,
        discover_capture_paths,
        estimate_reference_nfe_ms,
        split_capture_paths,
        split_capture_paths_by_metadata,
        split_grouped_capture_paths,
    )

    config = _load_config(args.config)
    validate_training_config(config)
    loss_weights = _resolve_loss_weights(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = _MetricLogger(
        output_dir=output_dir,
        config=config,
        run_name=output_dir.name,
    )

    capture_dirs = [args.capture_dir, *args.additional_capture_dir]
    capture_paths = tuple(
        dict.fromkeys(
            path for capture_dir in capture_dirs for path in discover_capture_paths(capture_dir)
        )
    )
    overfit_same_data = bool(config.get("overfit_same_data", False))
    split_metadata_key = config.get("split_metadata_key")
    overfit_capture_count = config.get("overfit_capture_count")
    if overfit_same_data:
        if args.group_by or split_metadata_key:
            raise ValueError("overfit_same_data cannot be combined with partition/group splits")
        if overfit_capture_count is None:
            raise ValueError("overfit_same_data requires overfit_capture_count")
        selected_count = int(overfit_capture_count)
        if not 1 <= selected_count <= len(capture_paths):
            raise ValueError("overfit_capture_count must be within available captures")
        selected = list(capture_paths)
        import random

        random.Random(seed).shuffle(selected)
        selected_paths = tuple(selected[:selected_count])
        capture_paths = selected_paths
        split = CaptureSplit(
            train=selected_paths,
            validation=selected_paths,
        )
    elif split_metadata_key:
        if args.group_by:
            raise ValueError("split_metadata_key cannot be combined with --group-by")
        split = split_capture_paths_by_metadata(
            capture_paths,
            field=str(split_metadata_key),
            train_value=str(config.get("split_train_value", "train")),
            validation_value=str(config.get("split_validation_value", "heldout")),
            group_by=config.get("split_group_by", "prompt"),
        )
    else:
        split_function = split_grouped_capture_paths if args.group_by else split_capture_paths
        split_kwargs = {
            "validation_fraction": float(config["validation_fraction"]),
            "seed": seed,
        }
        if args.group_by:
            split_kwargs["group_by"] = args.group_by
        split = split_function(capture_paths, **split_kwargs)
    split_record = {
        "strategy": (
            "same_data_overfit"
            if overfit_same_data
            else "metadata_fixed"
            if split_metadata_key
            else "grouped_random"
            if args.group_by
            else "capture_random"
        ),
        "group_by": (
            config.get("split_group_by", "prompt") if split_metadata_key else args.group_by
        ),
        "split_metadata_key": split_metadata_key,
        "seed": seed,
        "validation_fraction": float(config["validation_fraction"]),
        "train_captures": [path.name for path in split.train],
        "validation_captures": [path.name for path in split.validation],
    }
    (output_dir / "training_split.json").write_text(
        json.dumps(split_record, indent=2),
        encoding="utf-8",
    )
    measured_reference_nfe_ms = estimate_reference_nfe_ms(capture_paths)
    jump_horizons = tuple(sorted({int(value) for value in config.get("jump_horizons", [0])}))
    if not jump_horizons or min(jump_horizons) < 0:
        raise ValueError("jump_horizons must contain nonnegative integers")
    receipt_steps = config.get("receipt_steps")
    recurrent_max_event_index = config.get("recurrent_max_event_index")
    recurrent_max_rollout_age = config.get("recurrent_max_rollout_age")
    source_branch = str(config.get("source_branch", "old"))
    target_branch = str(config.get("target_branch", "new"))
    train_dataset = TransportPairDataset(
        split.train,
        jump_horizons=jump_horizons,
        receipt_steps=receipt_steps,
        source_branch=source_branch,
        target_branch=target_branch,
        recurrent_max_event_index=recurrent_max_event_index,
        recurrent_max_rollout_age=recurrent_max_rollout_age,
    )
    validation_dataset = TransportPairDataset(
        split.validation,
        jump_horizons=jump_horizons,
        receipt_steps=receipt_steps,
        source_branch=source_branch,
        target_branch=target_branch,
        recurrent_max_event_index=recurrent_max_event_index,
        recurrent_max_rollout_age=recurrent_max_rollout_age,
    )
    first_sample = train_dataset[0]
    inferred_latent_channels = int(first_sample["active_state"].shape[1])
    inferred_denoising_steps = int(first_sample["noise_trace"].shape[0]) + 1
    loader_options = {
        "num_workers": int(config["num_workers"]),
        "pin_memory": True,
    }
    recurrent_sequence_length = int(config.get("recurrent_sequence_length", 0))
    if recurrent_sequence_length < 0:
        raise ValueError("recurrent_sequence_length must be nonnegative")
    if recurrent_sequence_length:
        if int(config["batch_size"]) != 1:
            raise ValueError("Sequence-window training currently requires batch_size=1")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=RecurrentSequenceBatchSampler(
                train_dataset,
                sequence_length=recurrent_sequence_length,
                shuffle=True,
                seed=seed,
                exclude_scheduled_refresh=bool(
                    config.get("recurrent_exclude_scheduled_refresh", True)
                ),
            ),
            **loader_options,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=RecurrentSequenceBatchSampler(
                validation_dataset,
                sequence_length=recurrent_sequence_length,
                shuffle=False,
                seed=seed,
                exclude_scheduled_refresh=bool(
                    config.get("recurrent_exclude_scheduled_refresh", True)
                ),
            ),
            **loader_options,
        )
    else:
        balance_fields = tuple(config.get("balance_training_by", ()))
        if balance_fields:
            from collections import Counter

            from torch.utils.data import WeightedRandomSampler

            balance_keys = [
                train_dataset.balance_key(index, balance_fields)
                for index in range(len(train_dataset))
            ]
            balance_counts = Counter(balance_keys)
            train_sampler = WeightedRandomSampler(
                [1.0 / balance_counts[key] for key in balance_keys],
                num_samples=len(train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                sampler=train_sampler,
                drop_last=False,
                **loader_options,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                drop_last=False,
                **loader_options,
            )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            drop_last=False,
            **loader_options,
        )

    device = torch.device("cuda")
    model_config = TransportModelConfig(
        latent_channels=inferred_latent_channels,
        denoising_steps=inferred_denoising_steps,
        base_channels=int(config["base_channels"]),
        condition_channels=int(config["condition_channels"]),
        max_jump_horizon=max(jump_horizons),
        use_cached_prediction=bool(config.get("use_cached_prediction", False)),
        target_parameterization=str(config.get("target_parameterization", "state")),
        endpoint_noise_exclusive=bool(config.get("endpoint_noise_exclusive", True)),
        relative_pose_target_from_source=bool(config.get("relative_pose_target_from_source", True)),
        max_rollout_age=int(config.get("max_rollout_age", 0)),
        transport_role=str(config.get("transport_role", "generic")),
        use_temporal_suffix_mask=bool(config.get("use_temporal_suffix_mask", False)),
    )
    model = CounterfactualTransport(model_config).to(device)
    initialize_model_checkpoint = config.get("initialize_model_checkpoint")
    if initialize_model_checkpoint is not None:
        initialization = torch.load(
            Path(initialize_model_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(initialization["model"], strict=True)
    _validate_primary_transport_configuration(
        model_config=model_config,
        jump_horizons=jump_horizons,
        source_branch=source_branch,
        target_branch=target_branch,
    )
    perceptual_weights = {
        "lpips": loss_weights.lambda_lpips,
        "temporal": loss_weights.lambda_temporal,
        "boundary": loss_weights.lambda_history_boundary,
    }
    if any(weight < 0.0 for weight in perceptual_weights.values()):
        raise ValueError("Decoded perceptual loss weights must be nonnegative")
    decoded_perceptual = None
    if any(weight > 0.0 for weight in perceptual_weights.values()):
        decoded_perceptual = _load_decoded_perceptual_components(
            config=config,
            device=device,
        )
    max_steps = int(config["max_steps"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler_name = str(config.get("learning_rate_scheduler", "constant")).lower()
    if scheduler_name == "constant":
        lr_scheduler = None
    elif scheduler_name == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=float(config.get("minimum_learning_rate", 0.0)),
        )
    else:
        raise ValueError("learning_rate_scheduler must be 'constant' or 'cosine'")
    ema_decay = float(config.get("ema_decay", 0.0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    ema_model = deepcopy(model).requires_grad_(False) if ema_decay else None
    step = 0
    best_validation = math.inf
    best_validation_perceptual = math.inf
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("raw_model", payload["model"]))
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if lr_scheduler is not None and payload.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(payload["lr_scheduler"])
        step = int(payload.get("step", 0))
        best_validation = float(payload.get("validation_nmse", math.inf))
        if ema_model is not None and payload.get("ema_model") is not None:
            ema_model.load_state_dict(payload["ema_model"], strict=True)

    train_iterator = iter(train_loader)
    validation_interval = int(config["validation_interval"])
    checkpoint_interval = int(config.get("checkpoint_interval", validation_interval))
    metrics_log_interval = int(config.get("metrics_log_interval", 20))
    if validation_interval <= 0 or max_steps <= 0:
        raise ValueError("max_steps and validation_interval must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if metrics_log_interval <= 0:
        raise ValueError("metrics_log_interval must be positive")

    def latest_checkpoint(validation_nmse: float) -> dict[str, Any]:
        evaluation_model = ema_model if ema_model is not None else model
        payload = checkpoint_payload(
            evaluation_model,
            optimizer=optimizer,
            step=step,
            validation_nmse=validation_nmse,
        )
        if ema_model is not None:
            payload["raw_model"] = model.state_dict()
            payload["ema_model"] = ema_model.state_dict()
            payload["ema_decay"] = ema_decay
        if lr_scheduler is not None:
            payload["lr_scheduler"] = lr_scheduler.state_dict()
        payload["loss_weights"] = loss_weights.to_dict()
        return payload

    wall_started = time.perf_counter()
    history: list[dict[str, Any]] = []
    while step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = _move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(**_model_inputs(batch))
            effective_batch = batch
            loss, train_state_nmse, train_loss_terms = _training_losses(
                result,
                effective_batch,
                target_parameterization=model_config.target_parameterization,
                state_loss_weight=loss_weights.lambda_state,
                delta_loss_weight=loss_weights.lambda_residual,
                latent_gradient_loss_weight=float(config.get("latent_gradient_loss_weight", 0.0)),
                intra_chunk_boundary_loss_weight=loss_weights.lambda_mid,
            )
            decoded_loss_metrics: dict[str, Any] = {}
            if decoded_perceptual is not None:
                decoded_prediction, decoded_target, decoded_history = _decoded_loss_inputs(
                    prediction=result["predicted_target"],
                    batch=effective_batch,
                    endpoint_only=bool(
                        config.get(
                            "decoded_trajectory_endpoint_only",
                            False,
                        )
                    ),
                )
                decoded_loss, decoded_loss_metrics = _decoded_perceptual_losses(
                    prediction=decoded_prediction,
                    target=decoded_target,
                    history_tail=decoded_history,
                    decoder=decoded_perceptual["decoder"],
                    lpips_model=decoded_perceptual["lpips_model"],
                    lpips_weight=perceptual_weights["lpips"],
                    temporal_weight=perceptual_weights["temporal"],
                    boundary_weight=perceptual_weights["boundary"],
                    latent_crop_height=int(config.get("decoded_latent_crop_height", 30)),
                    latent_crop_width=int(config.get("decoded_latent_crop_width", 52)),
                    event_rgb_frames=int(config.get("decoded_event_rgb_frames", 16)),
                    lpips_frame_count=int(config.get("decoded_lpips_frame_count", 2)),
                    lpips_height=int(config.get("decoded_lpips_height", 120)),
                    lpips_width=int(config.get("decoded_lpips_width", 208)),
                    history_latent_frames=(
                        int(config["decoded_history_latent_frames"])
                        if config.get("decoded_history_latent_frames") is not None
                        else None
                    ),
                    prediction_latent_frames=(
                        int(config["decoded_prediction_latent_frames"])
                        if config.get("decoded_prediction_latent_frames") is not None
                        else None
                    ),
                )
                loss = loss + decoded_loss
        loss.backward()
        gradient_clip = float(config.get("gradient_clip", 1.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if ema_model is not None:
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema_parameter.mul_(ema_decay).add_(parameter.detach(), alpha=1.0 - ema_decay)
                for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
                    ema_buffer.copy_(buffer)
        step += 1

        if step == 1 or step % metrics_log_interval == 0:
            train_record = {
                "kind": "train",
                "step": step,
                "train_loss": float(loss.detach().cpu().item()),
                "train_reconstructed_state_nmse": float(train_state_nmse.detach().cpu().item()),
                **{
                    name: float(value.detach().cpu().item())
                    for name, value in train_loss_terms.items()
                },
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - wall_started,
                **{
                    name: float(value.detach().cpu().item())
                    for name, value in decoded_loss_metrics.items()
                },
            }
            metric_logger.log(train_record, step=step)

        validation_step = step == 1 or step % validation_interval == 0 or step == max_steps
        if step % checkpoint_interval == 0 and not validation_step:
            torch.save(latest_checkpoint(best_validation), output_dir / "latest.pt")

        if validation_step:
            evaluation_model = ema_model if ema_model is not None else model
            (
                validation_nmse,
                validation_clean_nmse,
                validation_by_horizon,
                validation_by_stratum,
                validation_by_intra_chunk_offset,
                validation_clean_by_intra_chunk_offset,
            ) = _evaluate(
                evaluation_model,
                validation_loader,
                device,
                target_parameterization=model_config.target_parameterization,
            )
            validation_decoded_metrics: dict[str, Any] = {}
            if decoded_perceptual is not None:
                validation_decoded_metrics = _evaluate_decoded_perceptual(
                    evaluation_model,
                    validation_loader,
                    device,
                    decoder=decoded_perceptual["decoder"],
                    lpips_model=decoded_perceptual["lpips_model"],
                    config=config,
                    weights=perceptual_weights,
                )
            record = {
                "kind": "validation",
                "step": step,
                "train_loss": float(loss.detach().cpu().item()),
                "train_reconstructed_state_nmse": float(train_state_nmse.detach().cpu().item()),
                **{
                    name: float(value.detach().cpu().item())
                    for name, value in train_loss_terms.items()
                },
                "validation_nmse": validation_nmse,
                "validation_clean_nmse": validation_clean_nmse,
                "validation_nmse_by_horizon": validation_by_horizon,
                "validation_nmse_by_stratum": validation_by_stratum,
                "validation_nmse_by_intra_chunk_offset": (validation_by_intra_chunk_offset),
                "validation_clean_nmse_by_intra_chunk_offset": (
                    validation_clean_by_intra_chunk_offset
                ),
                "validation_model": "ema" if ema_model is not None else "raw",
                **validation_decoded_metrics,
                **{
                    name: float(value.detach().cpu().item())
                    for name, value in decoded_loss_metrics.items()
                },
                "elapsed_seconds": time.perf_counter() - wall_started,
            }
            history.append(record)
            metric_logger.log(record, step=step)
            print(json.dumps(record), flush=True)
            latest = latest_checkpoint(validation_nmse)
            torch.save(latest, output_dir / "latest.pt")
            if validation_nmse < best_validation:
                best_validation = validation_nmse
                torch.save(latest, output_dir / "best.pt")
            validation_perceptual = validation_decoded_metrics.get(
                "validation_decoded_perceptual_weighted_loss"
            )
            max_perceptual_checkpoint_nmse = float(
                config.get("decoded_max_validation_nmse", math.inf)
            )
            if (
                validation_perceptual is not None
                and validation_nmse <= max_perceptual_checkpoint_nmse
                and validation_perceptual < best_validation_perceptual
            ):
                best_validation_perceptual = validation_perceptual
                torch.save(latest, output_dir / "best_perceptual.pt")

    final_raw_validation: dict[str, Any] | None = None
    if ema_model is not None:
        (
            raw_nmse,
            raw_clean_nmse,
            raw_by_horizon,
            raw_by_stratum,
            raw_by_intra_chunk_offset,
            raw_clean_by_intra_chunk_offset,
        ) = _evaluate(
            model,
            validation_loader,
            device,
            target_parameterization=model_config.target_parameterization,
        )
        final_raw_validation = {
            "validation_nmse": raw_nmse,
            "validation_clean_nmse": raw_clean_nmse,
            "validation_nmse_by_horizon": raw_by_horizon,
            "validation_nmse_by_stratum": raw_by_stratum,
            "validation_nmse_by_intra_chunk_offset": (raw_by_intra_chunk_offset),
            "validation_clean_nmse_by_intra_chunk_offset": (raw_clean_by_intra_chunk_offset),
        }
        (output_dir / "final_raw_validation.json").write_text(
            json.dumps(final_raw_validation, indent=2),
            encoding="utf-8",
        )

    benchmark = _benchmark(
        model,
        next(iter(validation_loader)),
        device,
        warmup=int(config.get("benchmark_warmup", 10)),
        repetitions=int(config.get("benchmark_repetitions", 50)),
    )
    benchmark["reference_nfe_ms"] = measured_reference_nfe_ms
    benchmark["corrector_nfe_equivalent"] = benchmark["median_ms"] / measured_reference_nfe_ms
    summary = {
        "status": "completed",
        "train_captures": len(split.train),
        "validation_captures": len(split.validation),
        "capture_dirs": [str(path.resolve()) for path in capture_dirs],
        "balance_training_by": list(config.get("balance_training_by", ())),
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset),
        "parameter_count": model_parameter_count(model),
        "latent_channels": inferred_latent_channels,
        "denoising_steps": inferred_denoising_steps,
        "jump_horizons": list(jump_horizons),
        "receipt_steps": (
            None if receipt_steps is None else [int(value) for value in receipt_steps]
        ),
        "recurrent_max_event_index": recurrent_max_event_index,
        "recurrent_max_rollout_age": recurrent_max_rollout_age,
        "recurrent_sequence_length": recurrent_sequence_length,
        "overfit_same_data": overfit_same_data,
        "target_parameterization": model_config.target_parameterization,
        "source_branch": source_branch,
        "target_branch": target_branch,
        "decoded_perceptual_enabled": decoded_perceptual is not None,
        "loss_weights": loss_weights.to_dict(),
        "ema_decay": ema_decay,
        "learning_rate_scheduler": scheduler_name,
        "initial_learning_rate": float(config["learning_rate"]),
        "minimum_learning_rate": float(config.get("minimum_learning_rate", 0.0)),
        "final_raw_validation": final_raw_validation,
        "best_validation_nmse": best_validation,
        "best_validation_perceptual_weighted_loss": (
            None if math.isinf(best_validation_perceptual) else best_validation_perceptual
        ),
        "required_best_validation_nmse": config.get("required_best_validation_nmse"),
        "diagnostic_threshold_passed": (
            None
            if config.get("required_best_validation_nmse") is None
            else best_validation < float(config["required_best_validation_nmse"])
        ),
        "steps": step,
        "wall_seconds": time.perf_counter() - wall_started,
        "benchmark": benchmark,
        "history": history,
        "config": config,
        "split_record": str(output_dir / "training_split.json"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    metric_logger.finish()
    print(json.dumps(summary, indent=2))


def _model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    names = {
        "active_state",
        "initial_state",
        "history_tail",
        "noise_trace",
        "old_viewmats",
        "new_viewmats",
        "intrinsics",
        "receipt_step",
        "jump_horizon",
        "cached_prediction",
        "temporal_suffix_mask",
    }
    inputs = {name: batch[name] for name in names if name in batch}
    if "rollout_age" in batch:
        inputs["rollout_age"] = batch["rollout_age"]
    return inputs


def _validate_primary_transport_configuration(
    *,
    model_config: Any,
    jump_horizons: tuple[int, ...],
    source_branch: str,
    target_branch: str,
) -> None:
    """Make the CST-R/CST-T operating points impossible to misconfigure."""
    role = str(model_config.transport_role)
    if role not in {
        "action_h0",
        "action_hm",
        "action_h0_state",
        "action_hm_state",
    }:
        return
    if source_branch != "old" or target_branch != "new":
        raise ValueError(f"{role} requires paired old-to-new action trajectories")
    if jump_horizons != (0,):
        raise ValueError(f"{role} training requires jump_horizons=[0]")
    if role in {"action_hm", "action_hm_state"} and not model_config.use_temporal_suffix_mask:
        raise ValueError(f"{role} requires use_temporal_suffix_mask=true")


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for name, value in batch.items()
    }


def _evaluate(
    model: Any,
    loader: Any,
    device: Any,
    *,
    target_parameterization: str,
) -> tuple[
    float,
    float | None,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    import torch

    from ..core.model import reconstruct_noisy_state

    model.eval()
    values: list[float] = []
    clean_values: list[float] = []
    by_horizon: dict[int, list[float]] = {}
    by_stratum: dict[str, list[float]] = {}
    by_intra_chunk_offset: dict[int, list[float]] = {}
    clean_by_intra_chunk_offset: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model(**_model_inputs(batch))
                effective_batch = batch
            prediction = result["predicted_target"].float()
            if target_parameterization == "clean_prediction":
                clean_target = effective_batch["clean_target"].float()
                clean_errors = _per_example_nmse(
                    prediction,
                    clean_target,
                    mask=effective_batch.get("temporal_suffix_mask"),
                )
                prediction = reconstruct_noisy_state(
                    prediction,
                    effective_batch["target_transition_noise"].float(),
                    effective_batch["target_sigma"].float(),
                )
                clean_values.extend(float(value) for value in clean_errors.detach().cpu().tolist())
            target = effective_batch["target_state"].float()
            errors = _per_example_nmse(
                prediction,
                target,
                mask=effective_batch.get("temporal_suffix_mask"),
            )
            offsets = batch.get("intra_chunk_offset")
            offset_values = (
                [None] * len(errors) if offsets is None else offsets.detach().cpu().tolist()
            )
            clean_error_values = (
                [None] * len(errors)
                if target_parameterization != "clean_prediction"
                else clean_errors.detach().cpu().tolist()
            )
            for error, clean_error, horizon, receipt, event, offset, transition in zip(
                errors.detach().cpu().tolist(),
                clean_error_values,
                batch["jump_horizon"].detach().cpu().tolist(),
                batch["receipt_step"].detach().cpu().tolist(),
                batch["recurrent_event_index"].detach().cpu().tolist(),
                offset_values,
                batch["action_transition"],
                strict=True,
            ):
                value = float(error)
                values.append(value)
                by_horizon.setdefault(int(horizon), []).append(value)
                if offset is None:
                    key = f"r{int(receipt)}|e{int(event)}|{transition}"
                else:
                    offset = int(offset)
                    key = f"r{int(receipt)}|m{offset}|e{int(event)}|{transition}"
                    by_intra_chunk_offset.setdefault(offset, []).append(value)
                    if clean_error is not None:
                        clean_by_intra_chunk_offset.setdefault(offset, []).append(
                            float(clean_error)
                        )
                by_stratum.setdefault(key, []).append(value)
    if not values:
        raise RuntimeError("Validation loader produced no batches")
    return (
        sum(values) / len(values),
        None if not clean_values else sum(clean_values) / len(clean_values),
        {
            str(horizon): sum(horizon_values) / len(horizon_values)
            for horizon, horizon_values in sorted(by_horizon.items())
        },
        {
            key: sum(stratum_values) / len(stratum_values)
            for key, stratum_values in sorted(by_stratum.items())
        },
        {
            str(offset): sum(offset_values) / len(offset_values)
            for offset, offset_values in sorted(by_intra_chunk_offset.items())
        },
        {
            str(offset): sum(offset_values) / len(offset_values)
            for offset, offset_values in sorted(clean_by_intra_chunk_offset.items())
        },
    )


def _evaluate_decoded_perceptual(
    model: Any,
    loader: Any,
    device: Any,
    *,
    decoder: Any,
    lpips_model: Any,
    config: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Evaluate decoded loss on the complete validation capture set."""
    import torch

    model.eval()
    values: dict[str, list[float]] = {}
    values_by_intra_chunk_offset: dict[int, dict[str, list[float]]] = {}
    batch_limit = int(config.get("decoded_validation_batch_limit", 0))
    if batch_limit < 0:
        raise ValueError("decoded_validation_batch_limit must be nonnegative")
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_limit and batch_index >= batch_limit:
                break
            batch = _move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model(**_model_inputs(batch))
                effective_batch = batch
                decoded_prediction, decoded_target, decoded_history = _decoded_loss_inputs(
                    prediction=result["predicted_target"],
                    batch=effective_batch,
                    endpoint_only=bool(
                        config.get(
                            "decoded_trajectory_endpoint_only",
                            False,
                        )
                    ),
                )
                _, metrics = _decoded_perceptual_losses(
                    prediction=decoded_prediction,
                    target=decoded_target,
                    history_tail=decoded_history,
                    decoder=decoder,
                    lpips_model=lpips_model,
                    lpips_weight=weights["lpips"],
                    temporal_weight=weights["temporal"],
                    boundary_weight=weights["boundary"],
                    latent_crop_height=int(config.get("decoded_latent_crop_height", 30)),
                    latent_crop_width=int(config.get("decoded_latent_crop_width", 52)),
                    event_rgb_frames=int(config.get("decoded_event_rgb_frames", 16)),
                    lpips_frame_count=int(config.get("decoded_lpips_frame_count", 2)),
                    lpips_height=int(config.get("decoded_lpips_height", 120)),
                    lpips_width=int(config.get("decoded_lpips_width", 208)),
                    history_latent_frames=(
                        int(config["decoded_history_latent_frames"])
                        if config.get("decoded_history_latent_frames") is not None
                        else None
                    ),
                    prediction_latent_frames=(
                        int(config["decoded_prediction_latent_frames"])
                        if config.get("decoded_prediction_latent_frames") is not None
                        else None
                    ),
                )
            for name, value in metrics.items():
                values.setdefault(name, []).append(float(value.detach().cpu().item()))
            offsets = effective_batch.get("intra_chunk_offset")
            if offsets is not None:
                unique_offsets = sorted({int(value) for value in offsets.detach().cpu().tolist()})
                for offset in unique_offsets:
                    indices = (offsets == offset).nonzero(as_tuple=False).flatten()
                    if int(indices.numel()) == int(offsets.numel()):
                        offset_metrics = metrics
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, offset_metrics = _decoded_perceptual_losses(
                                prediction=decoded_prediction[indices],
                                target=decoded_target[indices],
                                history_tail=decoded_history[indices],
                                decoder=decoder,
                                lpips_model=lpips_model,
                                lpips_weight=weights["lpips"],
                                temporal_weight=weights["temporal"],
                                boundary_weight=weights["boundary"],
                                latent_crop_height=int(
                                    config.get("decoded_latent_crop_height", 30)
                                ),
                                latent_crop_width=int(config.get("decoded_latent_crop_width", 52)),
                                event_rgb_frames=int(config.get("decoded_event_rgb_frames", 16)),
                                lpips_frame_count=int(config.get("decoded_lpips_frame_count", 2)),
                                lpips_height=int(config.get("decoded_lpips_height", 120)),
                                lpips_width=int(config.get("decoded_lpips_width", 208)),
                                history_latent_frames=(
                                    int(config["decoded_history_latent_frames"])
                                    if config.get("decoded_history_latent_frames") is not None
                                    else None
                                ),
                                prediction_latent_frames=(
                                    int(config["decoded_prediction_latent_frames"])
                                    if config.get("decoded_prediction_latent_frames") is not None
                                    else None
                                ),
                            )
                    offset_values = values_by_intra_chunk_offset.setdefault(offset, {})
                    for name, value in offset_metrics.items():
                        offset_values.setdefault(name, []).append(
                            float(value.detach().cpu().item())
                        )
    if not values:
        raise RuntimeError("Decoded validation produced no batches")
    result: dict[str, Any] = {
        f"validation_{name}": sum(items) / len(items) for name, items in values.items()
    }
    result["validation_decoded_by_intra_chunk_offset"] = {
        str(offset): {
            name: sum(items) / len(items) for name, items in sorted(offset_values.items())
        }
        for offset, offset_values in sorted(values_by_intra_chunk_offset.items())
    }
    return result


def _training_losses(
    result: dict[str, Any],
    batch: dict[str, Any],
    *,
    target_parameterization: str,
    state_loss_weight: float = 1.0,
    delta_loss_weight: float = 0.0,
    latent_gradient_loss_weight: float = 0.0,
    intra_chunk_boundary_loss_weight: float = 0.0,
) -> tuple[Any, Any, dict[str, Any]]:
    from ..core.model import normalized_state_mse, reconstruct_noisy_state

    if not math.isfinite(state_loss_weight) or state_loss_weight < 0.0:
        raise ValueError("state_loss_weight must be finite and nonnegative")
    prediction = result["predicted_target"]
    if target_parameterization == "state":
        suffix_mask = batch.get("temporal_suffix_mask")
        state_loss = (
            _masked_normalized_state_mse(
                prediction,
                batch["target_state"],
                suffix_mask,
            )
            if suffix_mask is not None
            else normalized_state_mse(prediction, batch["target_state"])
        )
        loss = state_loss_weight * state_loss
        loss_terms = {
            "loss/state_nmse": state_loss,
            "loss/state_weighted": loss,
        }
        if delta_loss_weight < 0.0:
            raise ValueError("delta_loss_weight must be nonnegative")
        if delta_loss_weight > 0.0:
            target_delta = batch["target_state"] - batch["active_state"]
            delta_loss = (
                _masked_normalized_state_mse(
                    result["delta"],
                    target_delta,
                    suffix_mask,
                )
                if suffix_mask is not None
                else normalized_state_mse(result["delta"], target_delta)
            )
            loss = loss + delta_loss_weight * delta_loss
            loss_terms["loss/delta_nmse"] = delta_loss
            loss_terms["loss/delta_weighted"] = delta_loss_weight * delta_loss
        if intra_chunk_boundary_loss_weight < 0.0:
            raise ValueError("intra_chunk_boundary_loss_weight must be nonnegative")
        if intra_chunk_boundary_loss_weight > 0.0:
            offsets = batch.get("intra_chunk_offset")
            if offsets is None:
                raise ValueError("Intra-chunk boundary loss requires offsets")
            boundary_loss = _intra_chunk_boundary_nmse(
                prediction,
                batch["target_state"],
                offsets,
            )
            loss = loss + intra_chunk_boundary_loss_weight * boundary_loss
            loss_terms["loss/intra_chunk_boundary_nmse"] = boundary_loss
            loss_terms["loss/intra_chunk_boundary_weighted"] = (
                intra_chunk_boundary_loss_weight * boundary_loss
            )
        loss_terms["loss/transport_weighted_total"] = loss
        return (
            loss,
            state_loss,
            loss_terms,
        )
    if target_parameterization != "clean_prediction":
        raise ValueError(f"Unknown target parameterization: {target_parameterization}")
    suffix_mask = batch.get("temporal_suffix_mask")
    endpoint_loss = (
        _masked_normalized_state_mse(
            prediction,
            batch["clean_target"],
            suffix_mask,
        )
        if suffix_mask is not None
        else normalized_state_mse(prediction, batch["clean_target"])
    )
    loss = state_loss_weight * endpoint_loss
    loss_terms = {
        "loss/endpoint_nmse": endpoint_loss,
        "loss/endpoint_weighted": loss,
    }
    if delta_loss_weight < 0.0:
        raise ValueError("delta_loss_weight must be nonnegative")
    if delta_loss_weight > 0.0:
        target_delta = batch["clean_target"] - batch["cached_prediction"]
        delta_loss = (
            _masked_normalized_state_mse(
                result["delta"],
                target_delta,
                suffix_mask,
            )
            if suffix_mask is not None
            else normalized_state_mse(result["delta"], target_delta)
        )
        loss = loss + delta_loss_weight * delta_loss
        loss_terms["loss/delta_nmse"] = delta_loss
        loss_terms["loss/delta_weighted"] = delta_loss_weight * delta_loss
    if latent_gradient_loss_weight < 0.0:
        raise ValueError("latent_gradient_loss_weight must be nonnegative")
    if latent_gradient_loss_weight > 0.0:
        latent_gradient_loss = _latent_gradient_nmse(
            prediction,
            batch["clean_target"],
        )
        loss = loss + latent_gradient_loss_weight * latent_gradient_loss
        loss_terms["loss/latent_gradient_nmse"] = latent_gradient_loss
        loss_terms["loss/latent_gradient_weighted"] = (
            latent_gradient_loss_weight * latent_gradient_loss
        )
    if intra_chunk_boundary_loss_weight < 0.0:
        raise ValueError("intra_chunk_boundary_loss_weight must be nonnegative")
    if intra_chunk_boundary_loss_weight > 0.0:
        offsets = batch.get("intra_chunk_offset")
        if offsets is None:
            raise ValueError("Intra-chunk boundary loss requires offsets")
        boundary_loss = _intra_chunk_boundary_nmse(
            prediction,
            batch["clean_target"],
            offsets,
        )
        loss = loss + intra_chunk_boundary_loss_weight * boundary_loss
        loss_terms["loss/intra_chunk_boundary_nmse"] = boundary_loss
        loss_terms["loss/intra_chunk_boundary_weighted"] = (
            intra_chunk_boundary_loss_weight * boundary_loss
        )
    reconstructed = reconstruct_noisy_state(
        prediction,
        batch["target_transition_noise"],
        batch["target_sigma"],
    )
    state_nmse = (
        _masked_normalized_state_mse(
            reconstructed,
            batch["target_state"],
            suffix_mask,
        )
        if suffix_mask is not None
        else normalized_state_mse(reconstructed, batch["target_state"])
    )
    loss_terms["loss/reconstructed_state_nmse"] = state_nmse
    loss_terms["loss/transport_weighted_total"] = loss
    return loss, state_nmse, loss_terms


def _masked_normalized_state_mse(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    epsilon: float = 1e-8,
) -> Any:
    """Normalize state error using only the editable temporal suffix."""
    import torch
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    if mask is None or mask.ndim != prediction.ndim:
        raise ValueError("Suffix mask must align with batched latent states")
    expanded = mask.to(device=prediction.device, dtype=torch.float32).expand_as(prediction)
    dimensions = tuple(range(1, prediction.ndim))
    counts = expanded.sum(dim=dimensions).clamp_min(1.0)
    numerator = ((prediction.float() - target.float()).square() * expanded).sum(
        dim=dimensions
    ) / counts
    denominator = (target.float().square() * expanded).sum(dim=dimensions) / counts
    return (numerator / denominator.clamp_min(epsilon)).mean()


def _intra_chunk_boundary_nmse(
    prediction: Any,
    target: Any,
    offsets: Any,
    *,
    epsilon: float = 1e-8,
) -> Any:
    """Match the latent transition immediately across each action boundary."""
    import torch

    losses = []
    for sample_index, offset_value in enumerate(offsets):
        offset = int(offset_value.detach().cpu().item())
        if not 0 < offset < int(prediction.shape[1]):
            raise ValueError("Intra-chunk offset is outside the latent chunk")
        predicted_delta = (
            prediction[sample_index, offset] - prediction[sample_index, offset - 1]
        ).float()
        target_delta = (target[sample_index, offset] - target[sample_index, offset - 1]).float()
        numerator = (predicted_delta - target_delta).square().mean()
        denominator = target_delta.square().mean().clamp_min(epsilon)
        losses.append(numerator / denominator)
    return torch.stack(losses).mean()


def _latent_gradient_nmse(prediction: Any, target: Any) -> Any:
    """Preserve temporal and spatial latent detail during rollout training."""
    losses = []
    for dimension in (-4, -2, -1):
        if int(prediction.shape[dimension]) <= 1:
            continue
        predicted_gradient = prediction.diff(dim=dimension).float()
        target_gradient = target.diff(dim=dimension).float()
        numerator = (predicted_gradient - target_gradient).square().mean()
        denominator = target_gradient.square().mean().clamp_min(1e-8)
        losses.append(numerator / denominator)
    if not losses:
        return prediction.float().new_zeros(())
    return sum(losses) / len(losses)


def _load_decoded_perceptual_components(
    *,
    config: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Load the frozen backbone-specific VAE and pretrained LPIPS network."""
    import torch

    decoded_backbone = str(config.get("decoded_backbone", "minwm_wan"))
    if decoded_backbone == "official_hyworld15":
        components = _load_hyworld15_decoder(config=config, device=device)
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False)
        lpips_model.eval().requires_grad_(False).to(device=device)
        components["lpips_model"] = lpips_model
        return components
    if decoded_backbone != "minwm_wan":
        raise ValueError(f"Unknown decoded_backbone {decoded_backbone!r}")

    minwm_root = Path(
        config.get(
            "decoded_minwm_root",
            os.environ.get("ACTIONSPLICE_MINWM_ROOT", "third_party/minWM"),
        )
    ).resolve()
    wan_root = minwm_root / "Wan21"
    for path in (minwm_root, wan_root):
        resolved = str(path)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    vae_checkpoint = Path(
        config.get(
            "decoded_vae_checkpoint",
            wan_root / "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        )
    ).resolve()
    if not vae_checkpoint.exists():
        raise FileNotFoundError(vae_checkpoint)
    from wan.modules.vae import _video_vae

    vae_model = (
        _video_vae(
            pretrained_path=str(vae_checkpoint),
            z_dim=16,
        )
        .eval()
        .requires_grad_(False)
        .to(device=device, dtype=torch.bfloat16)
    )
    mean = torch.tensor(
        [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    std = torch.tensor(
        [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        device=device,
        dtype=torch.bfloat16,
    )

    def decode(latents: Any) -> Any:
        scale = [mean, 1.0 / std]
        decoded = []
        for latent in latents.permute(0, 2, 1, 3, 4):
            pixels = vae_model.decode(latent.unsqueeze(0), scale)
            decoded.append(pixels.float().squeeze(0))
        return torch.stack(decoded).permute(0, 2, 1, 3, 4)

    import lpips

    lpips_model = lpips.LPIPS(net="alex", verbose=False)
    lpips_model.eval().requires_grad_(False).to(device=device)
    return {
        "decoder": decode,
        "lpips_model": lpips_model,
        "vae_model": vae_model,
    }


def _load_hyworld15_decoder(
    *,
    config: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Load the official HY-World 1.5 32-channel VAE for CST losses."""
    import torch

    root = Path(
        config.get(
            "decoded_hyworld_root",
            os.environ.get("ACTIONSPLICE_HYWORLD_ROOT", "third_party/HY-WorldPlay"),
        )
    ).resolve()
    model_path = Path(config["decoded_hyworld_model_path"]).resolve()
    if not root.exists() or not (model_path / "vae").exists():
        raise FileNotFoundError(
            f"HY-World 1.5 code/model path missing: {root}, {model_path / 'vae'}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import (
        AutoencoderKLConv3D,
    )

    vae_model = (
        AutoencoderKLConv3D.from_pretrained(
            model_path / "vae",
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .requires_grad_(False)
        .to(device=device)
    )

    def decode(latents: Any) -> Any:
        native = latents.permute(0, 2, 1, 3, 4).contiguous()
        if getattr(vae_model.config, "shift_factor", None):
            native = native / vae_model.config.scaling_factor + vae_model.config.shift_factor
        else:
            native = native / vae_model.config.scaling_factor
        pixels = vae_model.decode(native, return_dict=False)[0]
        return pixels.float().permute(0, 2, 1, 3, 4).contiguous()

    return {"decoder": decode, "vae_model": vae_model}


def _decoded_perceptual_losses(
    *,
    prediction: Any,
    target: Any,
    history_tail: Any,
    decoder: Any,
    lpips_model: Any,
    lpips_weight: float,
    temporal_weight: float,
    boundary_weight: float,
    latent_crop_height: int,
    latent_crop_width: int,
    event_rgb_frames: int,
    lpips_frame_count: int,
    lpips_height: int,
    lpips_width: int,
    history_latent_frames: int | None = None,
    prediction_latent_frames: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compare decoded terminal chunks while retaining causal history."""
    import torch
    import torch.nn.functional as functional

    if prediction.shape != target.shape or prediction.shape != history_tail.shape:
        raise ValueError("Decoded perceptual inputs must share latent shape")
    height, width = int(prediction.shape[-2]), int(prediction.shape[-1])
    if not 0 < latent_crop_height <= height:
        raise ValueError("decoded_latent_crop_height is outside latent height")
    if not 0 < latent_crop_width <= width:
        raise ValueError("decoded_latent_crop_width is outside latent width")
    if event_rgb_frames <= 1 or lpips_frame_count <= 0:
        raise ValueError("Decoded perceptual frame counts must be positive")
    top = (height - latent_crop_height) // 2
    left = (width - latent_crop_width) // 2

    def crop(latents: Any) -> Any:
        return latents[
            ...,
            top : top + latent_crop_height,
            left : left + latent_crop_width,
        ]

    history = crop(history_tail.detach())
    predicted = crop(prediction)
    target = crop(target.detach())
    if history_latent_frames is not None:
        if not 0 < history_latent_frames <= int(history.shape[1]):
            raise ValueError("decoded_history_latent_frames is outside history")
        history = history[:, -history_latent_frames:]
    if prediction_latent_frames is not None:
        if not 0 < prediction_latent_frames <= int(predicted.shape[1]):
            raise ValueError("decoded_prediction_latent_frames is outside prediction")
        predicted = predicted[:, :prediction_latent_frames]
        target = target[:, :prediction_latent_frames]
    predicted_context = torch.cat([history, predicted], dim=1)
    target_context = torch.cat([history, target], dim=1)
    with torch.no_grad():
        target_pixels = decoder(target_context).clamp(-1.0, 1.0)
    predicted_pixels = decoder(predicted_context).clamp(-1.0, 1.0)
    if int(predicted_pixels.shape[1]) <= event_rgb_frames:
        raise ValueError("Decoded context does not contain a preceding RGB frame")
    predicted_event = predicted_pixels[:, -event_rgb_frames:]
    target_event = target_pixels[:, -event_rgb_frames:]

    frame_indices = (
        torch.linspace(
            0,
            event_rgb_frames - 1,
            steps=min(lpips_frame_count, event_rgb_frames),
            device=prediction.device,
        )
        .round()
        .long()
        .unique()
    )
    predicted_lpips = predicted_event[:, frame_indices].flatten(0, 1)
    target_lpips = target_event[:, frame_indices].flatten(0, 1)
    predicted_lpips = functional.interpolate(
        predicted_lpips,
        size=(lpips_height, lpips_width),
        mode="bilinear",
        align_corners=False,
    )
    target_lpips = functional.interpolate(
        target_lpips,
        size=(lpips_height, lpips_width),
        mode="bilinear",
        align_corners=False,
    )
    lpips_loss = lpips_model(predicted_lpips, target_lpips).mean()
    temporal_loss = (predicted_event.diff(dim=1) - target_event.diff(dim=1)).abs().mean()
    predicted_boundary = predicted_event[:, 0] - predicted_pixels[:, -event_rgb_frames - 1]
    target_boundary = target_event[:, 0] - target_pixels[:, -event_rgb_frames - 1]
    boundary_loss = (predicted_boundary - target_boundary).abs().mean()
    total = (
        float(lpips_weight) * lpips_loss
        + float(temporal_weight) * temporal_loss
        + float(boundary_weight) * boundary_loss
    )
    return total, {
        "decoded_lpips_loss": lpips_loss,
        "decoded_temporal_loss": temporal_loss,
        "decoded_boundary_loss": boundary_loss,
        "decoded_perceptual_weighted_loss": total,
    }


def _decoded_loss_inputs(
    *,
    prediction: Any,
    batch: dict[str, Any],
    endpoint_only: bool,
) -> tuple[Any, Any, Any]:
    """Select the trajectory endpoint for memory-safe decoded supervision."""
    target = batch["clean_target"]
    history = batch["history_tail"]
    if not endpoint_only:
        return prediction, target, history
    event_indices = batch.get("recurrent_event_index")
    if event_indices is None:
        raise ValueError("Decoded trajectory endpoint selection requires event indices")
    endpoint_mask = event_indices == event_indices.max()
    return (
        prediction[endpoint_mask],
        target[endpoint_mask],
        history[endpoint_mask],
    )


def _per_example_nmse(
    prediction: Any,
    target: Any,
    *,
    mask: Any | None = None,
) -> Any:
    dimensions = tuple(range(1, prediction.ndim))
    if mask is not None:
        expanded = mask.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).expand_as(prediction)
        counts = expanded.sum(dim=dimensions).clamp_min(1.0)
        numerator = ((prediction.float() - target.float()).square() * expanded.float()).sum(
            dim=dimensions
        ) / counts.float()
        denominator = (target.float().square() * expanded.float()).sum(
            dim=dimensions
        ) / counts.float()
        return numerator / denominator.clamp_min(1e-8)
    return (prediction.float() - target.float()).square().mean(
        dim=dimensions
    ) / target.float().square().mean(dim=dimensions).clamp_min(1e-8)


def _benchmark(
    model: Any,
    batch: dict[str, Any],
    device: Any,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    import statistics

    import torch

    if warmup < 0 or repetitions <= 0:
        raise ValueError("Invalid benchmark iteration count")
    batch = _move_batch(batch, device)
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ),
    ):
        for _ in range(warmup):
            model(**_model_inputs(batch))
        torch.cuda.synchronize()
        elapsed: list[float] = []
        for _ in range(repetitions):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            model(**_model_inputs(batch))
            ended.record()
            ended.synchronize()
            elapsed.append(float(started.elapsed_time(ended)))
    return {
        "warmup": warmup,
        "repetitions": repetitions,
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.fmean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
    }



if __name__ == "__main__":
    main()
