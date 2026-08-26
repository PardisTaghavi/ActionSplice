"""Cache helpers used by the unmodified minWM matched-teacher capture."""

from __future__ import annotations

from typing import Any


def _reset_pipeline_caches(pipeline: Any, noise: Any) -> None:
    import torch

    batch_size = int(noise.shape[0])
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(batch_size, noise.dtype, noise.device)
        pipeline._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        pipeline._initialize_prope_kv_cache(batch_size, noise.dtype, noise.device)
        return
    for block in pipeline.crossattn_cache:
        block["is_init"] = False
    for cache in (pipeline.kv_cache1, pipeline.prope_kv_cache1):
        for block in cache:
            block["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
            block["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)


def _cache_index_snapshot(pipeline: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, cache in (("rope", pipeline.kv_cache1), ("prope", pipeline.prope_kv_cache1)):
        if not cache:
            raise RuntimeError(f"{name} cache is unavailable")
        block = cache[0]
        result[f"{name}_global_end"] = int(block["global_end_index"].item())
        result[f"{name}_local_end"] = int(block["local_end_index"].item())
    if (
        result["rope_global_end"] != result["prope_global_end"]
        or result["rope_local_end"] != result["prope_local_end"]
    ):
        raise AssertionError("RoPE and PRoPE cache indices diverged")
    return result


def validate_cache_overwrite_trace(
    events: list[dict[str, Any]], *, frame_seq_length: int | None = None
) -> dict[str, int | bool]:
    if frame_seq_length is not None and frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")
    expected_after_by_start: dict[int, dict[str, int]] = {}
    checked_repeated_calls = 0
    for event in events:
        before = event.get("cache_indices_before")
        after = event.get("cache_indices_after")
        if before is None or after is None:
            raise ValueError("Cache-index trace is missing from an event")
        start = int(event["start_frame"])
        if frame_seq_length is not None:
            expected_global_end = (start + int(event["frame_count"])) * frame_seq_length
            if (
                int(after["rope_global_end"]) != expected_global_end
                or int(after["prope_global_end"]) != expected_global_end
            ):
                raise AssertionError("Active-chunk cache ended at the wrong global token index")
        expected = expected_after_by_start.get(start)
        if expected is not None:
            checked_repeated_calls += 1
            if before != expected or after != expected:
                raise AssertionError("Repeated active-chunk call advanced cache indices")
        else:
            expected_after_by_start[start] = dict(after)
    return {
        "passed": True,
        "traced_event_count": len(events),
        "traced_chunk_count": len(expected_after_by_start),
        "checked_repeated_call_count": checked_repeated_calls,
    }
