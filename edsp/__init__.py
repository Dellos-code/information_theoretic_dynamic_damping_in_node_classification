"""Entropy-Damped Signed Propagation."""

from .core import (
    EDSPResult,
    edsp,
    entropy,
    heterophily_complement,
    kl_to_uniform,
    trust_calibrate,
)
from .torch_backend import edsp_torch

__all__ = [
    "EDSPResult",
    "edsp",
    "entropy",
    "heterophily_complement",
    "kl_to_uniform",
    "trust_calibrate",
    "edsp_torch",
]
