"""The single entry point that reads test labels (Master Prompt §2.7, rule R3).

PR-2 provides the access boundary and basic metrics; the full metric set of
§5.4 (macro-F1, balanced accuracy, ROC-AUC, AP, NLL, Brier, ECE) lands with the
evaluation PRs.
"""

from __future__ import annotations

import numpy as np

from edsp.data.vault import LabelVault


def evaluate(vault: LabelVault, probabilities: np.ndarray) -> dict:
    """Score node-level class probabilities (n, k) on the vault's test nodes."""
    idx, y = vault.test_labels()
    probs = np.asarray(probabilities)[idx]
    pred = probs.argmax(axis=1)
    eps = 1e-12
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))))
    return {"split_id": vault.split_id, "n_test": len(y), "accuracy": float(np.mean(pred == y)), "nll": nll}
