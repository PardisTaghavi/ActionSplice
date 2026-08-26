"""Thin shared backend registry plus upstream-specific adapters."""

from .base import BackendSpec, CSTMethod
from .registry import BACKENDS, get_backend, validate_training_config

__all__ = [
    "BACKENDS",
    "BackendSpec",
    "CSTMethod",
    "get_backend",
    "validate_training_config",
]
