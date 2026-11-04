"""Optional torch adapter for parity with the float64 reference backend."""

from __future__ import annotations

import numpy as np

from .core import EDSPResult, edsp


def edsp_torch(*args, **kwargs):
    """Run the reference algorithm and return tensors for torch callers.

    The reference backend remains the numerical authority in PR-1. This adapter
    provides an explicit parity surface without making torch a core dependency.
    """
    import torch

    result: EDSPResult = edsp(*args, **kwargs)
    return {
        "probabilities": torch.from_numpy(np.asarray(result.probabilities)),
        "predictions": torch.from_numpy(result.predictions),
        "p_esc": torch.from_numpy(np.asarray(result.p_esc)),
        "weights": torch.from_numpy(np.asarray(result.weights)),
        "iterations": result.iterations,
        "converged": result.converged,
        "final_residual": result.final_residual,
    }
