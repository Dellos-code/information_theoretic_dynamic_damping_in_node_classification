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


def edge_polarity_auc(vault: LabelVault, edges: np.ndarray, het_score: np.ndarray, threshold: float = 0.5) -> dict:
    """ROC-AUC of a P(heterophilic) edge score on test-test edges (H-LEARNED diagnostic)."""
    from edsp.utils.numerics import roc_auc

    idx, y = vault.test_labels()
    n = max(int(edges.max()) + 1 if len(edges) else 0, int(idx.max(initial=-1)) + 1)
    lab = np.full(n, -1, dtype=np.int64)
    lab[idx] = y
    u, v = edges[:, 0], edges[:, 1]
    mask = (lab[u] >= 0) & (lab[v] >= 0)
    target = lab[u[mask]] != lab[v[mask]]
    return {
        "split_id": vault.split_id,
        "test_test_edges": int(mask.sum()),
        "test_edge_heterophily": float(target.mean()) if mask.any() else None,
        "test_edge_auc": roc_auc(target, np.asarray(het_score)[mask]),
        "test_edge_accuracy": float(np.mean((np.asarray(het_score)[mask] >= threshold) == target)) if mask.any() else None,
        "threshold": threshold,
    }
