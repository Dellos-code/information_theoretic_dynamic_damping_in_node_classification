"""Edge-probability regimes p_ij (Master Prompt §2.2).

All functions return one probability per undirected edge of ``graph.edges``.
P-LEARNED is produced together with H-LEARNED in :mod:`edsp.regimes.polarity`.
"""

from __future__ import annotations

import numpy as np

from edsp.data.graph import GraphData

# STRING's prior probability of a random protein pair being linked; used by the
# combined-score formula. TO VERIFY against the STRING documentation (§2.2 iii).
STRING_PRIOR = 0.041


def p_const(graph: GraphData, p0: float = 0.5) -> np.ndarray:
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be in [0, 1]")
    return np.full(len(graph.edges), float(p0))


def _resolve_mu(values: np.ndarray, mu: float | str) -> float:
    if mu == "median":
        pos = values[values > 0]
        return float(np.median(pos)) if len(pos) else 1.0
    return float(mu)


def p_count(graph: GraphData, mu: float | str = "median", column: int = 0) -> np.ndarray:
    """P-COUNT: p = 1 - exp(-c / mu) for a non-negative edge count/weight c.

    Then -ln(1 - p) = c / mu, so the EDSP weights are proportional to c (D6).
    """
    if graph.edge_attr is None:
        raise ValueError(f"{graph.name}: P-COUNT needs edge counts in edge_attr")
    c = np.asarray(graph.edge_attr, dtype=np.float64).reshape(len(graph.edges), -1)[:, column]
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    return -np.expm1(-c / _resolve_mu(c, mu))


def p_signmag(graph: GraphData, mode: str = "exp", mu: float | str = "median") -> np.ndarray:
    """P-SIGNMAG for signed graphs: p = 1 - exp(-|w| / mu) or p = |w| / max|w|."""
    if graph.edge_sign is None:
        raise ValueError(f"{graph.name}: P-SIGNMAG needs edge signs")
    w = np.abs(np.asarray(graph.edge_sign, dtype=np.float64))
    if mode == "exp":
        return -np.expm1(-w / _resolve_mu(w, mu))
    if mode == "linear":
        top = w.max() if len(w) else 1.0
        return w / top if top > 0 else np.zeros_like(w)
    raise ValueError("mode must be 'exp' or 'linear'")


def p_native(graph: GraphData, mode: str = "noisy-or") -> np.ndarray:
    """P-NATIVE from per-channel confidence scores in [0, 1] (e.g. ogbn-proteins).

    modes: ``noisy-or`` (1 - prod(1 - s_c)), ``channel:<j>`` (one channel),
    ``string-combined`` (prior-corrected noisy-OR, STRING convention, TO VERIFY),
    ``max`` (max_c s_c) and ``mean`` (mean_c s_c) — the last two are the C4 substitutes
    for the single named channels while the ogbn-proteins column order is unverified
    (docs/SPEC_AMBIGUITIES.md, "ogbn-proteins edge-feature columns").

    Channels are processed one column at a time in float64 (C4: 39.5M edges x 8
    float32 channels would otherwise need several GB of float64 temporaries); the
    products are taken in channel order, as ``np.prod(..., axis=1)`` does.
    """
    if graph.edge_attr is None:
        raise ValueError(f"{graph.name}: P-NATIVE needs edge confidence scores")
    A = np.asarray(graph.edge_attr).reshape(len(graph.edges), -1)
    d = A.shape[1]

    def channel(c: int) -> np.ndarray:
        return np.clip(A[:, c].astype(np.float64), 0.0, 1.0)

    if mode == "noisy-or":
        q = np.ones(len(A))
        for c in range(d):
            q *= 1.0 - channel(c)
        return 1.0 - q
    if mode.startswith("channel:"):
        return channel(int(mode.split(":", 1)[1]))
    if mode == "max":
        out = channel(0)
        for c in range(1, d):
            np.maximum(out, channel(c), out=out)
        return out
    if mode == "mean":
        out = channel(0)
        for c in range(1, d):
            out += channel(c)
        return out / d
    if mode == "string-combined":
        q = np.ones(len(A))
        for c in range(d):
            q *= 1.0 - np.clip((channel(c) - STRING_PRIOR) / (1.0 - STRING_PRIOR), 0.0, 1.0)
        combined = 1.0 - q
        return combined + STRING_PRIOR * (1.0 - combined)
    raise ValueError(f"unknown P-NATIVE mode {mode!r}")
