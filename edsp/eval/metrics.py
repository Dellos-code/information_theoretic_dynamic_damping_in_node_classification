"""Metrics of Master Prompt §5.4 (pure functions; they never touch the vault).

Only edsp.eval.evaluate obtains test labels and passes them here.
"""

from __future__ import annotations

import numpy as np

from edsp.utils.numerics import roc_auc


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int, equal_mass: bool) -> float:
    if len(conf) == 0:
        return float("nan")
    if equal_mass:
        order = np.argsort(conf, kind="stable")
        groups = np.array_split(order, bins)
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
        groups = [np.flatnonzero(ids == b) for b in range(bins)]
    n = len(conf)
    return float(sum(len(g) / n * abs(correct[g].mean() - conf[g].mean()) for g in groups if len(g)))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="stable")
    hits = y_true[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / hits.sum())


def classification_metrics(y: np.ndarray, probs: np.ndarray, k: int) -> dict:
    y = np.asarray(y, dtype=np.int64)
    P = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    P = P / np.where(P.sum(1, keepdims=True) > 0, P.sum(1, keepdims=True), 1.0)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    conf = P.max(axis=1)
    f1s, recalls = [], []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    out = {
        "n_test": len(y),
        "accuracy": float(correct.mean()) if len(y) else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "nll": float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)))) if len(y) else float("nan"),
        "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))) if len(y) else float("nan"),
        "ece": _ece(conf, correct, 15, equal_mass=False),
        "adaptive_ece": _ece(conf, correct, 15, equal_mass=True),
    }
    if k == 2:
        out["roc_auc"] = roc_auc(y == 1, P[:, 1])
        out["average_precision"] = average_precision(y == 1, P[:, 1])
    else:
        aucs = [roc_auc(y == c, P[:, c]) for c in range(k)]
        aucs = [a for a in aucs if np.isfinite(a)]
        out["roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")  # one-vs-rest macro
    return out
