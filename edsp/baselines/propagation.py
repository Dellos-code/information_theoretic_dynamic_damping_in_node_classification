"""Implementations (numpy/scipy) of the B0/B1 baselines.

B0  majority, train-prior, feature-LR (the Track-B base predictor), signed-vote.
B1  harmonic LP (Zhu, Ghahramani & Lafferty 2003) with and without class mass
    normalisation, LGC (Zhou et al. 2004),
    PPR-LP (random-walk normalised LGC / training-free APPNP), Correct & Smooth
    (Huang et al. 2021) on the SAME base predictor as Track-B EDSP, LinBP with
    myopic compatibility estimation (Gatterbauer et al. 2015; Gatterbauer & Kumar
    2020 call it MCE), possible-worlds harmonic LP for probabilistic edges.

Graphs are used as undirected simple graphs (GraphData.edges); edge weights
default to 1 and can be set to edge probabilities with ``weights=...`` (harmonic,
harmonic_cmn, lgc, ppr_lp, linbp; suite C4 runs each of them also on the noisy-OR edge
probabilities of ogbn-proteins). ``weights=None`` is exactly the unweighted baseline.
Inside :func:`edsp.utils.memo.shared_structure` the label-free operators (adjacency,
its connected components and spectral radius) are built once per (edges, weights).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, eigsh, spsolve

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.regimes.priors import track_b_priors
from edsp.selection import select_by_val_metric
from edsp.utils.memo import memo

MAX_ITER = 1000
TOL = 1e-6


# ------------------------------------------------------------------ operators
def adjacency(graph: GraphData, weights: np.ndarray | None = None) -> sp.csr_matrix:
    """Symmetric (weighted) adjacency. Callers must not modify the returned matrix in
    place: inside ``shared_structure()`` it is shared between calls."""
    def build() -> sp.csr_matrix:
        e = graph.edges
        w = np.ones(len(e)) if weights is None else np.asarray(weights, dtype=np.float64)
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        return sp.csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(graph.num_nodes,) * 2)

    return memo(f"adjacency:{graph.num_nodes}", (graph.edges, weights), build)


def _random_walk(W: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return sp.diags(inv) @ W


def _sym_norm(W: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    inv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    D = sp.diags(inv)
    return D @ W @ D


def _one_hot_train(graph: GraphData, vault: LabelVault) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, y = vault.train_labels()
    Y = np.zeros((graph.num_nodes, graph.num_classes))
    Y[idx, y] = 1.0
    return Y, idx, y


def _to_probs(F: np.ndarray) -> np.ndarray:
    F = np.clip(F, 0.0, None)
    s = F.sum(axis=1, keepdims=True)
    k = F.shape[1]
    return np.divide(F, s, out=np.full_like(F, 1.0 / k), where=s > 0)


def _iterate(step: Callable[[np.ndarray], np.ndarray], F0: np.ndarray) -> tuple[np.ndarray, dict]:
    F = F0
    for it in range(1, MAX_ITER + 1):
        nxt = step(F)
        res = float(np.max(np.abs(nxt - F).sum(axis=1)))
        F = nxt
        if res < TOL:
            return F, {"iterations": it, "converged": True}
    return F, {"iterations": MAX_ITER, "converged": False, "final_residual": res}


def _solve(A: sp.csr_matrix, B: np.ndarray) -> tuple[np.ndarray, dict]:
    """Solve A X = B for a symmetric positive-definite sparse A.

    Preconditioned conjugate gradients (Jacobi preconditioner, rtol 1e-10) per
    column; a column that does not converge in 20,000 iterations falls back to an
    exact sparse LU solve. (PR-7b: exact LU for every system was the bottleneck on
    dense graphs such as tolokers — about 75 s per LGC solve.)"""
    if A.shape[0] == 0:
        return np.zeros_like(B), {"solver": "empty"}
    A = A.tocsr()
    d = A.diagonal()
    M = sp.diags(np.divide(1.0, d, out=np.ones_like(d), where=d != 0))
    cols, fallbacks = [], 0
    for j in range(B.shape[1]):
        b = B[:, j]
        if not np.any(b):
            cols.append(np.zeros_like(b))
            continue
        x, info = cg(A, b, rtol=1e-10, atol=0.0, maxiter=20_000, M=M)
        if info != 0:
            x = spsolve(A.tocsc(), b)
            fallbacks += 1
        cols.append(np.asarray(x))
    X = np.column_stack(cols)
    return X, {"solver": "pcg" if not fallbacks else f"pcg+lu({fallbacks})", "residual": float(np.abs(A @ X - B).max())}


# ---------------------------------------------------------------------- B0
def majority(graph: GraphData, vault: LabelVault) -> tuple[np.ndarray, dict]:
    _, y = vault.train_labels()
    c = int(np.bincount(y, minlength=graph.num_classes).argmax())
    P = np.zeros((graph.num_nodes, graph.num_classes))
    P[:, c] = 1.0
    return P, {"class": c}


def train_prior(graph: GraphData, vault: LabelVault) -> tuple[np.ndarray, dict]:
    _, y = vault.train_labels()
    pi = np.bincount(y, minlength=graph.num_classes) / len(y)
    return np.tile(pi, (graph.num_nodes, 1)), {}


def feature_lr(graph: GraphData, vault: LabelVault) -> tuple[np.ndarray, dict]:
    """The Track-B base predictor itself (logistic regression, validation-calibrated)."""
    v, _, info = track_b_priors(graph, vault)
    return v, info


def signed_vote(graph: GraphData, vault: LabelVault) -> tuple[np.ndarray, dict]:
    """score_c(j) = sum over train neighbours i of sign_ij * 1[y_i = c]; uncovered -> train prior."""
    if graph.edge_sign is None:
        raise ValueError("signed_vote needs edge signs")
    Y, _, y = _one_hot_train(graph, vault)
    S = adjacency(graph, np.sign(graph.edge_sign))
    score = S @ Y
    pi = np.bincount(y, minlength=graph.num_classes) / len(y)
    covered = np.abs(score).sum(axis=1) > 0
    P = np.tile(pi, (graph.num_nodes, 1))
    P[covered] = _to_probs(score[covered] - score[covered].min(axis=1, keepdims=True) + 1e-9)
    return P, {"covered_fraction": float(covered.mean())}


# ---------------------------------------------------------------------- B1
def harmonic(graph: GraphData, vault: LabelVault, weights: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Zhu, Ghahramani & Lafferty (2003): the harmonic solution f_u = (D_uu - W_uu)^-1 W_ul Y_l,
    solved exactly. Nodes in components without a labelled node (including isolated
    nodes) get the uniform distribution and are reported as uncovered."""
    Y, idx, _ = _one_hot_train(graph, vault)
    W = adjacency(graph, weights)
    n, k = graph.num_nodes, graph.num_classes
    _, comp = memo("components", (W,), lambda: connected_components(W, directed=False))
    labelled_comp = np.zeros(comp.max() + 1, dtype=bool)
    labelled_comp[comp[idx]] = True
    covered = labelled_comp[comp]
    is_lab = np.zeros(n, dtype=bool)
    is_lab[idx] = True
    U = np.flatnonzero(covered & ~is_lab)
    F = np.full((n, k), 1.0 / k)
    F[idx] = Y[idx]
    d = np.asarray(W.sum(axis=1)).ravel()
    A = (sp.diags(d) - W).tocsr()[U][:, U]
    B = W.tocsr()[U][:, idx] @ Y[idx]
    X, info = _solve(A, B)
    F[U] = X
    return _to_probs(F), info | {"uncovered_fraction": float(1 - covered.mean()), "converged": True}


def harmonic_cmn(graph: GraphData, vault: LabelVault, weights: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Harmonic function with class mass normalisation (Zhu et al. 2003, §3):
    column c of the unlabelled solution is rescaled so that its total mass is
    proportional to the train-class prior q_c."""
    F, info = harmonic(graph, vault, weights)
    idx, y = vault.train_labels()
    q = np.bincount(y, minlength=graph.num_classes) / len(y)
    unl = np.ones(graph.num_nodes, dtype=bool)
    unl[idx] = False
    mass = F[unl].sum(axis=0)
    scale = np.divide(q, mass, out=np.zeros_like(q), where=mass > 0)
    G = F.copy()
    G[unl] = F[unl] * scale
    return _to_probs(G), info | {"class_mass_normalisation": True}


def lgc(graph: GraphData, vault: LabelVault, alpha: float = 0.99, weights: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Zhou et al. (2004): F = (1 - alpha) (I - alpha S)^-1 Y, S = D^-1/2 W D^-1/2 (exact solve)."""
    Y, _, _ = _one_hot_train(graph, vault)
    S = _sym_norm(adjacency(graph, weights))
    F, info = _solve((sp.identity(graph.num_nodes) - alpha * S).tocsr(), (1 - alpha) * Y)
    return _to_probs(F), info | {"alpha": alpha, "converged": True}


def ppr_lp(graph: GraphData, vault: LabelVault, alpha: float = 0.9, weights: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Personalised-PageRank label propagation (training-free APPNP):
    F = (1 - alpha) (I - alpha P)^-1 Y, P = D^-1 W, solved as the symmetric
    positive-definite system (D - alpha W) F = (1 - alpha) D Y (isolated nodes keep F = (1 - alpha) Y)."""
    Y, _, _ = _one_hot_train(graph, vault)
    W = adjacency(graph, weights)
    d = np.asarray(W.sum(axis=1)).ravel()
    iso = d == 0
    A = (sp.diags(np.where(iso, 1.0, d)) - alpha * W).tocsr()
    B = (1 - alpha) * np.where(iso[:, None], Y, d[:, None] * Y)
    F, info = _solve(A, B)
    return _to_probs(F), info | {"alpha": alpha, "converged": True}


def signed_lgc(graph: GraphData, vault: LabelVault, alpha: float = 0.9) -> tuple[np.ndarray, dict]:
    """Signed-Laplacian label spreading (Kunegis et al., SDM 2010; the signed LP family
    compared in Mercado et al., ECML-PKDD 2019): F = (1 - alpha) (I - alpha S)^-1 Y_c with
    S = Dbar^-1/2 A_s Dbar^-1/2, A_s the summed signed adjacency, Dbar = diag(sum |A_s|),
    and Y_c the centred one-hot train labels (Y - 1/k on train rows), so a negative edge
    pushes its endpoint away from the neighbour's class. I - alpha S is positive definite
    for alpha < 1. Scores are mapped to probabilities by shifting each row to a zero
    minimum (as signed_vote); rows with no signal get the uniform distribution."""
    if graph.edge_sign is None:
        raise ValueError("signed_lgc needs edge signs")
    Y, idx, _ = _one_hot_train(graph, vault)
    k = graph.num_classes
    Yc = np.zeros_like(Y)
    Yc[idx] = Y[idx] - 1.0 / k
    A = adjacency(graph, graph.edge_sign)
    d = np.asarray(abs(A).sum(axis=1)).ravel()
    inv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    S = sp.diags(inv) @ A @ sp.diags(inv)
    F, info = _solve((sp.identity(graph.num_nodes) - alpha * S).tocsr(), (1 - alpha) * Yc)
    spread = F.max(axis=1) - F.min(axis=1)
    P = np.full_like(F, 1.0 / k)
    live = spread > 1e-12
    P[live] = _to_probs(F[live] - F[live].min(axis=1, keepdims=True) + 1e-9 * spread[live, None])
    return P, info | {"alpha": alpha, "converged": True, "no_signal_fraction": float(1 - live.mean())}


def correct_and_smooth(
    graph: GraphData, vault: LabelVault, alpha_correct: float = 0.8, alpha_smooth: float = 0.8, base: np.ndarray | None = None,
    hops: int = 1,
) -> tuple[np.ndarray, dict]:
    """Huang et al. (2021) with autoscale, on the Track-B base predictor.

    ``hops=2`` is the two-hop variant of Lim et al. (NeurIPS 2021; `edsp.baselines.posthoc.cs_2hop`):
    every propagation step applies S twice, S (S X), as ``num_hops=2`` in their
    `correct_smooth.general_outcome_correlation`; S^2 is never formed. ``hops=1`` (default) is
    the unchanged 1-hop C&S."""
    Z = feature_lr(graph, vault)[0] if base is None else base
    Y, idx, _ = _one_hot_train(graph, vault)
    S = _sym_norm(adjacency(graph))

    def prop(X: np.ndarray) -> np.ndarray:
        for _ in range(hops):
            X = S @ X
        return X

    E0 = np.zeros_like(Z)
    E0[idx] = Y[idx] - Z[idx]
    E, i1 = _iterate(lambda E: alpha_correct * prop(E) + (1 - alpha_correct) * E0, E0.copy())
    sigma = float(np.abs(E0[idx]).sum(axis=1).mean()) if len(idx) else 0.0
    norm = np.abs(E).sum(axis=1, keepdims=True)
    scale = np.divide(sigma, norm, out=np.zeros_like(norm), where=norm > 0)
    Zc = np.clip(Z + scale * E, 0.0, None)
    G0 = Zc.copy()
    G0[idx] = Y[idx]
    G, i2 = _iterate(lambda G: alpha_smooth * prop(G) + (1 - alpha_smooth) * G0, G0.copy())
    return _to_probs(G), {"alpha_correct": alpha_correct, "alpha_smooth": alpha_smooth, "sigma": sigma,
                          "correct": i1, "smooth": i2} | ({"hops": hops} if hops != 1 else {})


def estimate_compatibility(graph: GraphData, vault: LabelVault, smoothing: float = 1.0) -> np.ndarray:
    """Myopic compatibility estimation: row-normalised class counts on train-train edges, Laplace-smoothed."""
    idx, y = vault.train_labels()
    k = graph.num_classes
    lab = np.full(graph.num_nodes, -1)
    lab[idx] = y
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    m = (lab[u] >= 0) & (lab[v] >= 0)
    C = np.full((k, k), smoothing)
    np.add.at(C, (lab[u[m]], lab[v[m]]), 1.0)
    np.add.at(C, (lab[v[m]], lab[u[m]]), 1.0)
    return C / C.sum(axis=1, keepdims=True)


def _spectral_radius(W: sp.csr_matrix) -> float:
    return float(abs(eigsh(W.astype(np.float64), k=1, which="LM", return_eigenvectors=False,
                           v0=np.ones(W.shape[0]))[0])) if W.nnz else 1.0  # fixed v0: deterministic


def linbp(graph: GraphData, vault: LabelVault, scale: float = 0.5, H: np.ndarray | None = None,
          weights: np.ndarray | None = None, priors: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """LinBP* (no echo cancellation): F <- X + eps W F H_hat, H_hat = H - 1/k,
    eps = scale / (rho(W) rho(H_hat)) so the iteration converges for scale < 1.
    Beliefs are mapped to probabilities by adding 1/k, clipping at 0 and renormalising.
    ``weights`` (default None = unweighted) weight W; the compatibility estimate stays the
    myopic count on train-train edges. Explicit beliefs X: centred one-hot labels on train
    nodes, 0 elsewhere; ``priors`` (n, k) (LinBP-B, `edsp.baselines.posthoc.linbp_b`) sets the
    explicit beliefs of every non-train node to its centred prior, priors - 1/k."""
    k = graph.num_classes
    H = estimate_compatibility(graph, vault) if H is None else H
    Hc = H - 1.0 / k
    W = adjacency(graph, weights)
    rho_w = memo("spectral_radius", (W,), lambda: _spectral_radius(W))
    rho_h = float(np.max(np.abs(np.linalg.eigvals(Hc)))) or 1.0
    eps = scale / (rho_w * rho_h)
    Y, idx, _ = _one_hot_train(graph, vault)
    X = np.zeros_like(Y) if priors is None else np.asarray(priors, dtype=np.float64) - 1.0 / k
    X[idx] = Y[idx] - 1.0 / k
    F, info = _iterate(lambda F: X + eps * (W @ F) @ Hc, X.copy())
    return _to_probs(F + 1.0 / k), info | {"scale": scale, "eps": eps, "rho_W": rho_w, "rho_H": rho_h, "H": H.round(4).tolist()}


def possible_worlds_harmonic(graph: GraphData, vault: LabelVault, p: np.ndarray, samples: int = 10, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Average of harmonic LP over sampled worlds (edge kept with probability p_ij)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((graph.num_nodes, graph.num_classes))
    for _ in range(samples):
        keep = rng.random(len(graph.edges)) < p
        w = keep.astype(float)
        acc += harmonic(graph, vault, weights=w)[0]
    return acc / samples, {"samples": samples, "seed": seed}


# ----------------------------------------------------------- registry + selection
GRIDS: dict[str, dict[str, list]] = {
    # 9 points each, matching the 3 x 3 Correct & Smooth grid and the 9-config
    # EDSP-tuned grid (rule R6: equal tuning budgets).
    "lgc": {"alpha": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]},
    "ppr_lp": {"alpha": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]},
    "correct_and_smooth": {"alpha_correct": [0.5, 0.8, 0.95], "alpha_smooth": [0.5, 0.8, 0.95]},
    "signed_lgc": {"alpha": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]},
    "linbp": {"scale": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]},
}

BASELINES: dict[str, tuple[str, str, Callable]] = {
    # name: (group, track, function)
    "majority": ("B0", "A", majority),
    "train_prior": ("B0", "A", train_prior),
    "signed_vote": ("B0", "A", signed_vote),
    "feature_lr": ("B0", "B", feature_lr),
    "harmonic": ("B1", "A", harmonic),
    "harmonic_cmn": ("B1", "A", harmonic_cmn),
    "lgc": ("B1", "A", lgc),
    "ppr_lp": ("B1", "A", ppr_lp),
    "linbp": ("B1", "A", linbp),
    "signed_lgc": ("B1", "A", signed_lgc),
    "correct_and_smooth": ("B1", "B", correct_and_smooth),
}


WEIGHTED = ("harmonic", "harmonic_cmn", "lgc", "ppr_lp", "linbp")  # accept ``weights=``


def select_and_predict(name: str, graph: GraphData, vault: LabelVault, metric: str = "accuracy",
                       weights: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Run baseline ``name`` over its validation grid (GRIDS) and keep the best by
    the validation value of ``metric`` (edsp.selection). Grids are identical in size class to the
    EDSP tuning space; the zero-tuning setting is the function default. ``weights`` (one per
    undirected edge) runs the weighted version of a baseline in WEIGHTED; None (default) is
    the unweighted baseline, called exactly as before."""
    fn = BASELINES[name][2]
    grid = GRIDS.get(name)
    extra: dict = {}
    if weights is not None:
        if name not in WEIGHTED:
            raise ValueError(f"baseline {name!r} takes no edge weights")
        extra["weights"] = weights
    if not grid:
        return fn(graph, vault, **extra)
    if name == "correct_and_smooth":
        extra["base"] = feature_lr(graph, vault)[0]  # fit the base predictor once
    keys = list(grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]
    outs = [fn(graph, vault, **c, **extra) for c in combos]
    best, scores = select_by_val_metric(vault, [o[0] for o in outs], metric)
    probs, info = outs[best]
    return probs, info | {"grid": combos, "selection_metric": metric, "val_scores": scores, "chosen": combos[best]}
