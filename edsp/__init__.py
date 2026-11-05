"""Entropy-Damped Signed Propagation."""

from .core import (
    EDSPResult,
    canonical_phi,
    edsp,
    edsp_damped,
    edsp_gs,
    entropy,
    heterophily_complement,
    kl_to_uniform,
    trust_calibrate,
)
from .torch_backend import edsp_torch

__all__ = [
    "EDSPResult",
    "canonical_phi",
    "edsp",
    "edsp_damped",
    "edsp_gs",
    "edsp_torch",
    "entropy",
    "heterophily_complement",
    "kl_to_uniform",
    "trust_calibrate",
]
