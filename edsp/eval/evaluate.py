"""The single entry point that reads test labels (Master Prompt §2.7, rule R3).

Metrics: accuracy, macro-F1, balanced accuracy, ROC-AUC (binary; one-vs-rest
macro for k > 2), average precision (binary), NLL, Brier, ECE (15 equal-width
bins) and adaptive ECE (15 equal-mass bins) — see edsp.eval.metrics.
"""

from __future__ import annotations

import numpy as np

from edsp.data.vault import LabelVault


def evaluate(vault: LabelVault, probabilities: np.ndarray) -> dict:
    """Score node-level class probabilities (n, k) on the vault's test nodes
    with the full metric set of Master Prompt §5.4."""
    from edsp.eval.metrics import classification_metrics

    idx, y = vault.test_labels()
    probs = np.asarray(probabilities)[idx]
    return {"split_id": vault.split_id, **classification_metrics(y, probs, vault.num_classes)}


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


def stratified_accuracy(vault: LabelVault, probabilities: np.ndarray, bins: np.ndarray, n_bins: int = 5) -> dict:
    """Test accuracy per local-homophily bin (Master Prompt §5.7): {"count": [...], "correct": [...]}."""
    idx, y = vault.test_labels()
    pred = np.asarray(probabilities)[idx].argmax(axis=1)
    b = np.asarray(bins)[idx]
    count = [int(np.sum(b == i)) for i in range(n_bins)]
    correct = [int(np.sum((b == i) & (pred == y))) for i in range(n_bins)]
    return {"n_bins": n_bins, "count": count, "correct": correct}
