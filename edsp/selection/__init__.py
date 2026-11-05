"""Model selection: may read validation labels (rule R3), never test labels.

Validation labels never leave this module: callers pass candidate predictions
and receive the index of the chosen candidate plus validation scores.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_softmax

from edsp.data.vault import LabelVault
from edsp.utils.numerics import fit_temperature, roc_auc


def validation_accuracy(vault: LabelVault, probabilities: np.ndarray) -> float:
    idx, y = vault.val_labels()
    return float(np.mean(np.asarray(probabilities)[idx].argmax(axis=1) == y))


def _train_val_labels(vault: LabelVault, n: int) -> tuple[np.ndarray, np.ndarray]:
    tr_idx, tr_y = vault.train_labels()
    va_idx, va_y = vault.val_labels()
    y = np.full(n, -1, dtype=np.int64)
    y[tr_idx] = tr_y
    y[va_idx] = va_y
    is_val = np.zeros(n, dtype=bool)
    is_val[va_idx] = True
    return y, is_val


def select_by_val_edge_auc(vault: LabelVault, edges: np.ndarray, candidate_scores: list[np.ndarray]) -> tuple[int, list[float]]:
    """Choose the candidate P(heterophilic) scores with the best ROC-AUC on
    train-val and val-val edges (Master Prompt §2.3c)."""
    n = int(edges.max()) + 1 if len(edges) else 0
    n = max(n, int(max(vault.train_idx.max(initial=-1), vault.val_idx.max(initial=-1))) + 1)
    y, is_val = _train_val_labels(vault, n)
    u, v = edges[:, 0], edges[:, 1]
    mask = (y[u] >= 0) & (y[v] >= 0) & (is_val[u] | is_val[v])
    target = y[u[mask]] != y[v[mask]]
    aucs = [roc_auc(target, np.asarray(s)[mask]) for s in candidate_scores]
    finite = [a if np.isfinite(a) else -np.inf for a in aucs]
    return int(np.argmax(finite)), aucs


def select_threshold_by_val_edge_accuracy(
    vault: LabelVault, edges: np.ndarray, score: np.ndarray, thresholds: tuple[float, ...]
) -> tuple[float, list[float]]:
    """Choose the decision threshold on a P(heterophilic) score by accuracy on
    train-val and val-val edges; ties go to the threshold closest to 0.5."""
    n = int(edges.max()) + 1 if len(edges) else 0
    n = max(n, int(max(vault.train_idx.max(initial=-1), vault.val_idx.max(initial=-1))) + 1)
    y, is_val = _train_val_labels(vault, n)
    u, v = edges[:, 0], edges[:, 1]
    mask = (y[u] >= 0) & (y[v] >= 0) & (is_val[u] | is_val[v])
    if not mask.any():
        return 0.5, [float("nan")] * len(thresholds)
    target = y[u[mask]] != y[v[mask]]
    s = np.asarray(score)[mask]
    accs = [float(np.mean((s >= t) == target)) for t in thresholds]
    best = max(range(len(thresholds)), key=lambda i: (accs[i], -abs(thresholds[i] - 0.5)))
    return float(thresholds[best]), accs


def validation_score(vault: LabelVault, probabilities: np.ndarray, metric: str = "accuracy") -> float:
    """Validation value of a §5.4 metric (higher is better); NaN without validation nodes."""
    from edsp.eval.metrics import classification_metrics

    idx, y = vault.val_labels()
    if len(idx) == 0:
        return float("nan")
    return float(classification_metrics(y, np.asarray(probabilities)[idx], vault.num_classes)[metric])


def select_by_val_metric(vault: LabelVault, candidate_probs: list[np.ndarray], metric: str = "accuracy") -> tuple[int, list[float]]:
    scores = [validation_score(vault, p, metric) for p in candidate_probs]
    finite = [s if np.isfinite(s) else -np.inf for s in scores]
    return int(np.argmax(finite)), scores


def select_by_val_accuracy(vault: LabelVault, candidate_probs: list[np.ndarray]) -> tuple[int, list[float]]:
    accs = [validation_accuracy(vault, p) for p in candidate_probs]
    return int(np.argmax(accs)), accs


def val_temperature(vault: LabelVault, logits: np.ndarray) -> float:
    idx, y = vault.val_labels()
    return fit_temperature(np.asarray(logits)[idx], y)


def val_nll(vault: LabelVault, logits: np.ndarray, temperature: float = 1.0) -> float:
    idx, y = vault.val_labels()
    lp = log_softmax(np.asarray(logits)[idx] / temperature, axis=1)
    return float(-np.mean(lp[np.arange(len(y)), y]))
