"""Literal EDSP-canonical NumPy reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1e-12
_BOUNDARY = 1e-15


def _as_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    values = np.maximum(values, 0.0)
    sums = values.sum(axis=-1, keepdims=True)
    return np.divide(values, sums, out=np.full_like(values, 1.0 / values.shape[-1]), where=sums > 0)


def entropy(values: np.ndarray, *, base: float = math.e) -> np.ndarray:
    """Shannon entropy, with zero terms defined as zero."""
    p = _as_probabilities(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)
    result = -terms.sum(axis=-1)
    if base != math.e:
        result /= math.log(base)
    return result


def kl_to_uniform(values: np.ndarray, *, base: float = math.e) -> np.ndarray:
    """KL(P || uniform), equal to Hmax - H(P)."""
    p = _as_probabilities(values)
    k = p.shape[-1]
    return math.log(k, base) - entropy(p, base=base)


def heterophily_complement(values: np.ndarray) -> np.ndarray:
    """Apply C(P)=(1-P)/(k-1) row-wise."""
    p = _as_probabilities(values)
    k = p.shape[-1]
    if k < 2:
        raise ValueError("k must be at least 2")
    return (1.0 - p) / (k - 1)


def trust_calibrate(
    priors: np.ndarray,
    probabilities: np.ndarray,
    prior_trust: np.ndarray | float,
    edge_trust: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the one-time trust calibration and return (v, p, clipped_count)."""
    v = _as_probabilities(priors)
    p = np.asarray(probabilities, dtype=np.float64)
    prior_trust = np.asarray(prior_trust, dtype=np.float64)
    edge_trust = np.asarray(edge_trust, dtype=np.float64)
    v = (1.0 - np.exp(-prior_trust)[..., None]) * v + np.exp(-prior_trust)[..., None] / v.shape[-1]
    p = p * (1.0 - np.exp(-edge_trust))
    clipped = int(np.count_nonzero((p < 0.0) | (p >= 1.0)))
    p = np.clip(p, 0.0, 1.0 - 1e-12)
    return v, p, clipped


@dataclass(frozen=True)
class EDSPResult:
    probabilities: np.ndarray
    iterations: int
    converged: bool
    final_residual: float
    residual_trace: tuple[float, ...]
    alpha_trace: tuple[np.ndarray, ...]
    p_esc: np.ndarray
    weights: np.ndarray
    clipped_probability_count: int
    oscillation: bool

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.probabilities, axis=1)


def _directed_edges(
    n: int,
    edges: np.ndarray,
    probabilities: np.ndarray,
    heterophily: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int8),
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2)")
    p = np.asarray(probabilities, dtype=np.float64)
    h = np.asarray(heterophily, dtype=np.int8)
    if len(p) != len(edges) or len(h) != len(edges):
        raise ValueError("edge arrays must have equal length")
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    dp = np.concatenate((p, p))
    dh = np.concatenate((h, h))
    order = np.argsort(directed[:, 1], kind="stable")
    return directed[order], dp[order], dh[order]


def edsp(
    edges: Iterable[tuple[int, int]] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    heterophily: Iterable[int] | np.ndarray,
    priors: np.ndarray,
    *,
    prior_trust: np.ndarray | float = np.inf,
    edge_trust: np.ndarray | float = np.inf,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> EDSPResult:
    """Run synchronous Jacobi EDSP exactly as specified in §1.1–§1.5."""
    v, p, clipped = trust_calibrate(priors, probabilities, prior_trust, edge_trust)
    n, k = v.shape
    edges, p, h = _directed_edges(n, np.asarray(edges), p, np.asarray(heterophily))
    source, target = edges[:, 0], edges[:, 1]
    log_odds = -np.log1p(-p)
    den = np.bincount(target, weights=log_odds, minlength=n)
    weights = np.divide(log_odds, den[target], out=np.zeros_like(log_odds), where=den[target] > 0)
    p_esc = -np.expm1(-den)
    base_entropy = entropy(v)
    hmax = math.log(k)
    current = v.copy()
    residual_trace: list[float] = []
    alpha_trace: list[np.ndarray] = []
    delta_trace: list[np.ndarray] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        transformed = np.where(
            h[:, None] == 0,
            current[source],
            (1.0 - current[source]) / (k - 1),
        )
        aggregate = np.zeros_like(current)
        np.add.at(aggregate, target, weights[:, None] * transformed)
        aggregate[den == 0] = 1.0 / k
        aggregate_entropy = entropy(aggregate)
        numerator = (hmax - aggregate_entropy) * p_esc
        denominator = numerator + (hmax - base_entropy)
        alpha = np.divide(numerator, denominator, out=np.zeros(n), where=denominator > _BOUNDARY)
        boundary = (
            ((np.abs(hmax - base_entropy) <= _EPS * hmax) & (np.abs(hmax - aggregate_entropy) <= _EPS * hmax))
            | ((p_esc == 0) & (np.abs(hmax - base_entropy) <= _EPS * hmax))
        )
        next_values = (1.0 - alpha[:, None]) * v + alpha[:, None] * aggregate
        next_values[boundary] = 1.0 / k
        next_values = np.maximum(next_values, 0.0)
        next_values /= next_values.sum(axis=1, keepdims=True)
        residual = float(np.max(np.sum(np.abs(next_values - current), axis=1)))
        residual_trace.append(residual)
        alpha_trace.append(alpha.copy())
        delta_trace.append(next_values - current)
        current = next_values
        if residual < epsilon:
            converged = True
            break

    oscillation = False
    if len(residual_trace) >= 20:
        recent = np.asarray(residual_trace[-20:])
        alternating = np.sum(np.sign(np.asarray(delta_trace[-1]) * np.asarray(delta_trace[-2])) < 0)
        oscillation = bool(np.all(np.diff(recent) >= 0) and alternating >= max(1, n // 100))
    return EDSPResult(
        probabilities=current,
        iterations=len(residual_trace),
        converged=converged,
        final_residual=residual_trace[-1] if residual_trace else 0.0,
        residual_trace=tuple(residual_trace),
        alpha_trace=tuple(alpha_trace),
        p_esc=p_esc,
        weights=weights,
        clipped_probability_count=clipped,
        oscillation=oscillation,
    )
