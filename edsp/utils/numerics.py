"""Logistic regression, randomized SVD, standardisation and ROC-AUC.

Implemented with numpy/scipy so that the locked environment needs no
scikit-learn. Logistic regression minimises the mean (optionally class-balanced)
softmax cross-entropy plus ``l2 / 2 * ||W||^2`` with L-BFGS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from scipy.stats import rankdata


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> Standardizer:
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean, scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


@dataclass
class LogReg:
    W: np.ndarray  # (d, k)
    b: np.ndarray  # (k,)
    scaler: Standardizer

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(X, dtype=np.float64)) @ self.W + self.b

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.logits(X) / temperature, axis=1)


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    l2: float = 1.0,
    balanced: bool = True,
    max_iter: int = 300,
) -> LogReg:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    scaler = Standardizer.fit(X)
    Z = scaler.transform(X)
    n, d = Z.shape
    if balanced:
        counts = np.bincount(y, minlength=k).astype(float)
        cw = np.divide(n, k * counts, out=np.zeros(k), where=counts > 0)
        w = cw[y]
    else:
        w = np.ones(n)
    w = w / w.sum()
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1.0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        W = theta[: d * k].reshape(d, k)
        b = theta[d * k :]
        logits = Z @ W + b
        lp = log_softmax(logits, axis=1)
        loss = -np.sum(w * lp[np.arange(n), y]) + 0.5 * l2 * np.sum(W * W)
        G = (np.exp(lp) - Y) * w[:, None]
        gW = Z.T @ G + l2 * W
        gb = G.sum(axis=0)
        return float(loss), np.concatenate([gW.ravel(), gb])

    res = minimize(objective, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    theta = res.x
    return LogReg(theta[: d * k].reshape(d, k), theta[d * k :], scaler)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising NLL of softmax(logits / T) (Guo et al. 2017)."""

    def nll(log_t: np.ndarray) -> float:
        lp = log_softmax(logits / np.exp(log_t[0]), axis=1)
        return float(-np.mean(lp[np.arange(len(y)), y]))

    res = minimize(nll, np.zeros(1), method="L-BFGS-B", bounds=[(-4.0, 4.0)])
    return float(np.exp(res.x[0]))


def randomized_svd(X: np.ndarray, rank: int, *, n_iter: int = 4, seed: int = 0) -> np.ndarray:
    """Left factor U * S of a rank-``rank`` randomized SVD (Halko et al. 2011)."""
    rng = np.random.default_rng(seed)
    rank = min(rank, min(X.shape) - 1) if min(X.shape) > 1 else 1
    Q = np.linalg.qr(X @ rng.standard_normal((X.shape[1], rank + 8)))[0]
    for _ in range(n_iter):
        Q = np.linalg.qr(X.T @ Q)[0]
        Q = np.linalg.qr(X @ Q)[0]
    B = np.asarray((X.T @ Q).T)
    Ub, s, _ = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub[:, :rank]) * s[:rank]


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties; NaN if one class is absent."""
    y_true = np.asarray(y_true).astype(bool)
    pos, neg = int(y_true.sum()), int((~y_true).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[y_true].sum() - pos * (pos + 1) / 2) / (pos * neg))
