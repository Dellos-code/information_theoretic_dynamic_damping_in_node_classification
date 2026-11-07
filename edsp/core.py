"""Literal EDSP-canonical NumPy reference implementation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

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


def _regimes_from_entropies(base_entropy: np.ndarray, aggregate_entropy: np.ndarray, hmax: float) -> np.ndarray:
    tol = _EPS * hmax
    h_is_max = np.abs(hmax - base_entropy) <= tol
    hp_is_max = np.abs(hmax - aggregate_entropy) <= tol
    regime = np.full(base_entropy.shape[0], 4, dtype=np.int8)
    regime[h_is_max & hp_is_max] = 1
    regime[h_is_max & ~hp_is_max] = 2
    regime[~h_is_max & hp_is_max] = 3
    return regime


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
    alpha_histograms: dict[int, np.ndarray]
    regime_trace: tuple[np.ndarray, ...]
    entropy_trace: tuple[float, ...]
    p_esc: np.ndarray
    weights: np.ndarray
    clipped_probability_count: int
    oscillation: bool
    oscillation_period: int | None = None
    oscillation_amplitude: float = 0.0
    last_iterates: tuple[np.ndarray, np.ndarray] | None = None
    last_alpha: np.ndarray | None = None
    phi_residual_l1: float = 0.0

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


def _phi_step(
    current: np.ndarray,
    v: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    h: np.ndarray,
    weights: np.ndarray,
    den: np.ndarray,
    p_esc: np.ndarray,
    base_entropy: np.ndarray,
    hmax: float,
    k: int,
    operator: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    het = (1.0 - current[source]) / (k - 1) if operator is None else current[source] @ operator
    transformed = np.where(h[:, None] == 0, current[source], het)
    aggregate = np.zeros_like(current)
    np.add.at(aggregate, target, weights[:, None] * transformed)
    aggregate[den == 0] = 1.0 / k
    aggregate_entropy = entropy(aggregate)
    numerator = (hmax - aggregate_entropy) * p_esc
    denominator = numerator + (hmax - base_entropy)
    alpha = np.divide(numerator, denominator, out=np.zeros(current.shape[0]), where=denominator > _BOUNDARY)
    boundary = (
        ((np.abs(hmax - base_entropy) <= _EPS * hmax) & (np.abs(hmax - aggregate_entropy) <= _EPS * hmax))
        | ((p_esc == 0) & (np.abs(hmax - base_entropy) <= _EPS * hmax))
    )
    next_values = (1.0 - alpha[:, None]) * v + alpha[:, None] * aggregate
    next_values[boundary] = 1.0 / k
    next_values = np.maximum(next_values, 0.0)
    next_values /= next_values.sum(axis=1, keepdims=True)
    regime = _regimes_from_entropies(base_entropy, aggregate_entropy, hmax)
    return next_values, alpha, regime


def canonical_phi(
    edges: Iterable[tuple[int, int]] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    heterophily: Iterable[int] | np.ndarray,
    priors: np.ndarray,
    state: np.ndarray,
    *,
    prior_trust: np.ndarray | float = np.inf,
    edge_trust: np.ndarray | float = np.inf,
) -> np.ndarray:
    """Apply one canonical Jacobi update Φ to a supplied state."""
    v, p, _ = trust_calibrate(priors, np.asarray(probabilities), prior_trust, edge_trust)
    n, k = v.shape
    edges, p, h = _directed_edges(n, np.asarray(edges), p, np.asarray(heterophily))
    source, target = edges[:, 0], edges[:, 1]
    log_odds = -np.log1p(-p)
    den = np.bincount(target, weights=log_odds, minlength=n)
    weights = np.divide(log_odds, den[target], out=np.zeros_like(log_odds), where=den[target] > 0)
    p_esc = -np.expm1(-den)
    base_entropy = entropy(v)
    hmax = math.log(k)
    next_values, _, _ = _phi_step(
        _as_probabilities(np.asarray(state)),
        v,
        source,
        target,
        h,
        weights,
        den,
        p_esc,
        base_entropy,
        hmax,
        k,
    )
    return next_values


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
    variant: str = "EDSP-canonical",
    beta: float = 1.0,
    order: str = "id",
    seed: int | None = None,
    heterophilic_operator: np.ndarray | None = None,
) -> EDSPResult:
    """Run canonical EDSP or one of the explicitly named iteration variants.

    ``heterophilic_operator`` (named variant EDSP-fullH, Master Prompt §2.5): a k×k
    row-stochastic matrix M applied to heterophilic messages as V M instead of the
    canonical complement C(V) = V (J − I)/(k − 1). None (default) keeps the canonical
    complement; canonical EDSP itself is never altered (R5)."""
    if variant not in {"EDSP-canonical", "EDSP-GS", "EDSP-damped"}:
        raise ValueError(f"unknown EDSP variant: {variant}")
    if variant == "EDSP-canonical" and not np.isclose(beta, 1.0):
        raise ValueError("EDSP-canonical requires beta=1")
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if order not in {"id", "random"}:
        raise ValueError("order must be 'id' or 'random'")
    v, p, clipped = trust_calibrate(priors, np.asarray(probabilities), prior_trust, edge_trust)
    n, k = v.shape
    M = None
    if heterophilic_operator is not None:
        M = np.asarray(heterophilic_operator, dtype=np.float64)
        if M.shape != (k, k) or np.any(M < 0) or not np.allclose(M.sum(axis=1), 1.0):
            raise ValueError("heterophilic_operator must be a k x k row-stochastic matrix")
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
    regime_trace: list[np.ndarray] = []
    entropy_trace: list[float] = []
    alpha_samples: dict[int, np.ndarray] = {}
    converged = False
    rng = np.random.default_rng(seed)
    oscillation = False
    oscillation_amplitude = 0.0
    previous_previous: np.ndarray | None = None
    counts = np.bincount(target, minlength=n)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    for iteration in range(1, max_iter + 1):
        if variant == "EDSP-GS":
            update_order = np.arange(n) if order == "id" else rng.permutation(n)
            next_values = current.copy()
            alpha = np.zeros(n)
            aggregate_entropy = np.zeros(n)
            for target_node in update_order:
                start = offsets[target_node]
                end = offsets[target_node + 1]
                incoming = slice(start, end)
                if end > start:
                    src_vals = next_values[source[incoming]]
                    het_vals = (1.0 - src_vals) / (k - 1) if M is None else src_vals @ M
                    transformed = np.where(h[incoming, None] == 0, src_vals, het_vals)
                    aggregate_node = np.sum(weights[incoming, None] * transformed, axis=0)
                else:
                    aggregate_node = np.full(k, 1.0 / k)
                aggregate_entropy_node = float(entropy(aggregate_node)[0])
                aggregate_entropy[target_node] = aggregate_entropy_node
                numerator_node = (hmax - aggregate_entropy_node) * p_esc[target_node]
                denominator_node = numerator_node + (hmax - base_entropy[target_node])
                alpha_node = 0.0 if denominator_node <= _BOUNDARY else float(numerator_node / denominator_node)
                alpha[target_node] = alpha_node
                boundary_node = (
                    (abs(hmax - base_entropy[target_node]) <= _EPS * hmax
                     and abs(hmax - aggregate_entropy_node) <= _EPS * hmax)
                    or (p_esc[target_node] == 0 and abs(hmax - base_entropy[target_node]) <= _EPS * hmax)
                )
                value = (1 - alpha_node) * v[target_node] + alpha_node * aggregate_node
                if boundary_node:
                    next_values[target_node] = np.full(k, 1.0 / k)
                else:
                    value = np.maximum(value, 0.0)
                    total = float(value.sum())
                    next_values[target_node] = value / total if total > 0 else np.full(k, 1.0 / k)
            regime = _regimes_from_entropies(base_entropy, aggregate_entropy, hmax)
        else:
            proposed, alpha, regime = _phi_step(
                current,
                v,
                source,
                target,
                h,
                weights,
                den,
                p_esc,
                base_entropy,
                hmax,
                k,
                M,
            )
            next_values = (1 - beta) * current + beta * proposed
            next_values = np.maximum(next_values, 0.0)
            next_values /= next_values.sum(axis=1, keepdims=True)
        residual = float(np.max(np.sum(np.abs(next_values - current), axis=1)))
        if previous_previous is not None and iteration <= max_iter and iteration >= 3:
            period_two_distance = float(np.max(np.sum(np.abs(next_values - previous_previous), axis=1)))
            adjacent_distance = float(np.max(np.sum(np.abs(next_values - current), axis=1)))
            if period_two_distance < 1e-9 and adjacent_distance >= epsilon:
                oscillation = True
                oscillation_amplitude = max(oscillation_amplitude, adjacent_distance)
        residual_trace.append(residual)
        entropy_trace.append(float(np.mean(entropy(next_values))))
        regime_trace.append(np.bincount(regime, minlength=5)[1:] / n)
        if iteration in {1, 5}:
            alpha_samples[iteration] = alpha.copy()
        alpha_samples[-1] = alpha.copy()
        previous_previous, current = current, next_values
        if residual < epsilon:
            converged = True
            break

    phi_next, _, _ = _phi_step(current, v, source, target, h, weights, den, p_esc, base_entropy, hmax, k, M)

    return EDSPResult(
        probabilities=current,
        iterations=len(residual_trace),
        converged=converged,
        final_residual=residual_trace[-1] if residual_trace else 0.0,
        residual_trace=tuple(residual_trace),
        alpha_histograms={
            (len(residual_trace) if i == -1 else i): np.histogram(values, bins=20, range=(0.0, 1.0))[0]
            for i, values in alpha_samples.items()
        },
        regime_trace=tuple(regime_trace),
        entropy_trace=tuple(entropy_trace),
        p_esc=p_esc,
        weights=weights,
        clipped_probability_count=clipped,
        oscillation=oscillation,
        oscillation_period=2 if oscillation else None,
        oscillation_amplitude=oscillation_amplitude,
        last_iterates=(previous_previous if previous_previous is not None else v.copy(), current.copy()),
        last_alpha=alpha.copy(),
        phi_residual_l1=float(np.max(np.sum(np.abs(phi_next - current), axis=1))),
    )


def edsp_gs(*args, order: str = "id", seed: int | None = None, **kwargs) -> EDSPResult:
    return edsp(*args, variant="EDSP-GS", order=order, seed=seed, **kwargs)


def edsp_damped(*args, beta: float = 0.5, **kwargs) -> EDSPResult:
    return edsp(*args, variant="EDSP-damped", beta=beta, **kwargs)
