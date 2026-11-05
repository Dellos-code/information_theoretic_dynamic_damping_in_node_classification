"""Statistical comparisons of Master Prompt §5.9."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_resamples, len(v)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (NaN if fewer than 2 non-zero differences)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if np.count_nonzero(d) < 2:
        return float("nan")
    return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def friedman_nemenyi(scores: np.ndarray, alpha: float = 0.05) -> dict:
    """scores: (n_datasets, n_methods), higher is better. Returns average ranks
    (1 = best), the Friedman p-value and the Nemenyi critical difference."""
    S = np.asarray(scores, float)
    n, k = S.shape
    ranks = np.vstack([stats.rankdata(-row) for row in S])
    avg = ranks.mean(axis=0)
    p = float(stats.friedmanchisquare(*S.T).pvalue) if n >= 2 and k >= 3 else float("nan")
    q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * n))
    return {"average_ranks": avg.tolist(), "friedman_p": p, "nemenyi_cd": cd, "n_datasets": n}
