"""scipy.sparse (CSR) backend of the synchronous EDSP iteration (Master Prompt §1.7).

Same algorithm, same float64 arithmetic and same outputs/diagnostics as the reference
:func:`edsp.core.edsp` for the Jacobi variants (``EDSP-canonical``, ``EDSP-damped``,
optional ``heterophilic_operator`` of EDSP-fullH); parity is tested to 1e-12 in
tests/unit/test_core.py. Only the neighbour aggregation differs: the reference
accumulates per directed edge with ``np.add.at``; here it is one CSR product

    S = A @ [V ; C(V)],   A[j, i + n h_ij] = w_ij,

whose entries are stored in the reference's edge order (directed edges sorted stably by
target), so each S_j is summed over the same terms in the same order. Measured on random
graphs with n = 132,534 and k = 2 (docs/SPEC_AMBIGUITIES.md, "C4 per-task protocol"): at
2M edges 0.051 s per iteration against 0.36 s for the reference; at ogbn-proteins' 39.56M
edges 0.34 s per iteration, where the reference's per-edge temporaries would need ~10 GB.

:class:`DirectedPlan` caches the label-free part (the target ordering of the directed
edges) so that the many EDSP runs of suite C4 on one edge set sort it once.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from .core import EDSPResult, _update_from_aggregate, entropy, trust_calibrate


class DirectedPlan:
    """Directed-edge ordering of an undirected edge list (both directions, sorted
    stably by target, exactly as edsp.core._directed_edges)."""

    def __init__(self, num_nodes: int, edges: np.ndarray) -> None:
        edges = np.asarray(edges)
        self.num_nodes = int(num_nodes)
        self.num_edges = len(edges)
        self._edges = edges
        idx = np.int32 if 2 * max(self.num_edges, self.num_nodes) < np.iinfo(np.int32).max else np.int64
        target = np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64, copy=False)
        self.order = np.argsort(target, kind="stable").astype(idx)
        self.target = target[self.order].astype(idx)
        del target
        self.source = np.concatenate((edges[:, 0], edges[:, 1]))[self.order].astype(idx)
        counts = np.bincount(self.target, minlength=self.num_nodes)
        self.indptr = np.zeros(self.num_nodes + 1, dtype=idx)
        np.cumsum(counts, out=self.indptr[1:])

    def matches(self, num_nodes: int, edges: np.ndarray) -> bool:
        edges = np.asarray(edges)
        return (int(num_nodes) == self.num_nodes and edges.shape == self._edges.shape
                and (edges is self._edges or np.array_equal(edges, self._edges)))

    def directed(self, per_edge: np.ndarray) -> np.ndarray:
        per_edge = np.asarray(per_edge)
        return np.concatenate((per_edge, per_edge))[self.order]


def edsp_csr(
    edges: np.ndarray,
    probabilities: np.ndarray,
    heterophily: np.ndarray,
    priors: np.ndarray,
    *,
    prior_trust: np.ndarray | float = np.inf,
    edge_trust: np.ndarray | float = np.inf,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
    variant: str = "EDSP-canonical",
    beta: float = 1.0,
    heterophilic_operator: np.ndarray | None = None,
    plan: DirectedPlan | None = None,
) -> EDSPResult:
    """CSR backend of edsp.core.edsp for the Jacobi variants (no Gauss-Seidel)."""
    if variant not in {"EDSP-canonical", "EDSP-damped"}:
        raise ValueError(f"the CSR backend runs EDSP-canonical and EDSP-damped only, not {variant}")
    if variant == "EDSP-canonical" and not np.isclose(beta, 1.0):
        raise ValueError("EDSP-canonical requires beta=1")
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    v, p, clipped = trust_calibrate(priors, np.asarray(probabilities), prior_trust, edge_trust)
    n, k = v.shape
    M = None
    if heterophilic_operator is not None:
        M = np.asarray(heterophilic_operator, dtype=np.float64)
        if M.shape != (k, k) or np.any(M < 0) or not np.allclose(M.sum(axis=1), 1.0):
            raise ValueError("heterophilic_operator must be a k x k row-stochastic matrix")
    edges = np.asarray(edges)
    if edges.size and (edges.ndim != 2 or edges.shape[1] != 2):
        raise ValueError("edges must have shape (m, 2)")
    h_in = np.asarray(heterophily, dtype=np.int8)
    if len(p) != len(edges) or len(h_in) != len(edges):
        raise ValueError("edge arrays must have equal length")
    if plan is None or not plan.matches(n, edges):
        plan = DirectedPlan(n, edges.reshape(-1, 2))
    dp = plan.directed(p)
    log_odds = -np.log1p(-dp)
    del dp
    den = np.bincount(plan.target, weights=log_odds, minlength=n)
    den_t = den[plan.target]
    weights = np.divide(log_odds, den_t, out=np.zeros_like(log_odds), where=den_t > 0)
    del log_odds, den_t
    dh = plan.directed(h_in)
    columns = plan.source.astype(np.int64) + n * (dh != 0)
    del dh
    A = sp.csr_matrix((weights, columns.astype(plan.source.dtype), plan.indptr), shape=(n, 2 * n))
    del columns
    p_esc = -np.expm1(-den)
    base_entropy = entropy(v)
    hmax = math.log(k)

    def step(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        het = (1.0 - current) / (k - 1) if M is None else current @ M
        aggregate = np.asarray(A @ np.vstack((current, het)))
        return _update_from_aggregate(aggregate, v, den, p_esc, base_entropy, hmax, k)

    current = v.copy()
    residual_trace: list[float] = []
    regime_trace: list[np.ndarray] = []
    entropy_trace: list[float] = []
    alpha_samples: dict[int, np.ndarray] = {}
    converged = False
    oscillation = False
    oscillation_amplitude = 0.0
    previous_previous: np.ndarray | None = None
    alpha = np.zeros(n)
    for iteration in range(1, max_iter + 1):
        proposed, alpha, regime = step(current)
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

    phi_next, _, _ = step(current)
    weights_ref = A.data  # per directed edge, in the reference (target-sorted) order
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
        weights=weights_ref,
        clipped_probability_count=clipped,
        oscillation=oscillation,
        oscillation_period=2 if oscillation else None,
        oscillation_amplitude=oscillation_amplitude,
        last_iterates=(previous_previous if previous_previous is not None else v.copy(), current.copy()),
        last_alpha=alpha.copy(),
        phi_residual_l1=float(np.max(np.sum(np.abs(phi_next - current), axis=1))),
    )
