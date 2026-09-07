"""Thin, external adapter around an unmodified minWM checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _prepend_minwm_paths(minwm_root: Path) -> None:
    for path in (minwm_root, minwm_root / "Wan21", minwm_root / "shared"):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def minwm_commit(minwm_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=minwm_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_wan_pipeline(
    minwm_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    low_memory: bool | None = None,
) -> tuple[Any, Any, Any]:
    """Load minWM's four-step Wan pipeline and validate all checkpoint keys."""
    import torch
    from omegaconf import OmegaConf

    minwm_root = minwm_root.resolve()
    _prepend_minwm_paths(minwm_root)
    os.chdir(minwm_root)

    from pipeline import CausalInferencePipeline

    config = OmegaConf.load(str(config_path))
    default_config = OmegaConf.load(str(minwm_root / "Wan21/configs/default_config.yaml"))
    config = OmegaConf.merge(default_config, config)
    if len(config.denoising_step_list) != 4:
        raise ValueError("ActionSplice requires a four-step minWM checkpoint/configuration")

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.set_grad_enabled(False)
    free_vram_gb = torch.cuda.mem_get_info(device)[0] / (1024**3)
    use_low_memory = free_vram_gb < 45.0 if low_memory is None else low_memory
    pipeline = CausalInferencePipeline(config, device=device)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator_state = state_dict.get("generator_ema", state_dict.get("generator"))
    if generator_state is None:
        raise KeyError("Checkpoint contains neither generator_ema nor generator")
    normalized_state = {
        key.replace("model._fsdp_wrapped_module.", "model.", 1)
        if key.startswith("model._fsdp_wrapped_module.")
        else key: value
        for key, value in generator_state.items()
    }
    if len(normalized_state) != len(generator_state):
        raise ValueError("Checkpoint key normalization produced duplicate keys")
    pipeline.generator.load_state_dict(normalized_state, strict=True)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if use_low_memory:
        from demo_utils import memory as memory_utils

        memory_utils.gpu = device
        memory_utils.DynamicSwapInstaller.install_model(
            pipeline.text_encoder,
            device=device,
        )
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    prompt_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    prompt_cache_stats = {"hits": 0, "misses": 0}
    original_text_forward = pipeline.text_encoder.forward

    def cached_text_forward(text_prompts: list[str]) -> dict[str, Any]:
        key = tuple(str(prompt) for prompt in text_prompts)
        cached = prompt_cache.get(key)
        if cached is not None:
            prompt_cache_stats["hits"] += 1
            return cached
        prompt_cache_stats["misses"] += 1
        encoded = original_text_forward(text_prompts=text_prompts)
        prompt_cache[key] = encoded
        return encoded

    pipeline.text_encoder.forward = cached_text_forward
    pipeline.reactive_low_memory = use_low_memory
    pipeline.reactive_initial_free_vram_gb = free_vram_gb
    pipeline.reactive_prompt_cache = prompt_cache
    pipeline.reactive_prompt_cache_stats = prompt_cache_stats
    resident_vram_gb = torch.cuda.memory_allocated(device) / (1024**3)
    print(
        "ActionSplice minWM loader: "
        f"device={torch.cuda.get_device_name(device)}, "
        f"initial_free_vram_gb={free_vram_gb:.2f}, "
        f"low_memory={use_low_memory}, "
        f"resident_cuda_memory_gb={resident_vram_gb:.2f}"
    )
    return pipeline, config, device


__all__ = ["load_wan_pipeline", "minwm_commit"]
