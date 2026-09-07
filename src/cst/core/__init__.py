"""Backend-independent CST model, state adapters, and same-step runtime.

The model symbols are lazy so registry/config tooling does not import PyTorch.
"""

from typing import Any

from .state import HY15_ACTION2V_STATE_SPEC, WAN_ACTION2V_STATE_SPEC


def __getattr__(name: str) -> Any:
    if name in {"CounterfactualTransport", "TransportModelConfig"}:
        from .model import CounterfactualTransport, TransportModelConfig

        return {
            "CounterfactualTransport": CounterfactualTransport,
            "TransportModelConfig": TransportModelConfig,
        }[name]
    raise AttributeError(name)


__all__ = [
    "CounterfactualTransport",
    "HY15_ACTION2V_STATE_SPEC",
    "TransportModelConfig",
    "WAN_ACTION2V_STATE_SPEC",
]
