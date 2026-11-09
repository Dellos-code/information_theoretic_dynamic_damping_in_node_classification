"""Track-B post-hoc comparators on the SAME base predictor as EDSP-B and C&S (PR-39).

Each method refines the soft outputs Z of ``feature_lr`` (the Track-B base predictor:
logistic regression on node features, validation-calibrated; `edsp.regimes.priors`), reads
train labels only (``vault.train_labels()``) and is validation-selected over a 9-point grid
(rule R6: the budget of C&S and EDSP-tuned). Sources, grids and deviations:
docs/SPEC_AMBIGUITIES.md (PR-39 rows) and docs/BASELINES.md.

cs_2hop  C&S with two-hop propagation (Lim et al., NeurIPS 2021, arXiv 2110.14446; code
         CUAI/Non-Homophily-Large-Scale@82f8f05, `correct_smooth.py`): the repository's
         C&S (`propagation.correct_and_smooth`) with every propagation step applying the
         normalised adjacency S = D^-1/2 A D^-1/2 twice, S (S X) — their ``num_hops=2``.
         S^2 (or A^2) is never formed, so memory equals 1-hop C&S.
clp      Compatible Label Propagation (Zhong, Ivanov & Pang, TMLR 2022, arXiv 2205.09389;
         code zhiqiangzhongddu/TMLR-CLP@2f7901d, `label_prop_heterophily.py`, `utils.py`,
         `Model-CLP-Tog.ipynb`): Sinkhorn-normalised compatibility H from train labels and
         base predictions, per-class compatibility edge weights, 50 propagation layers with
         L1 row normalisation, restart and clamp to [0, 1].
linbp_b  LinBP (Gatterbauer et al., VLDB 2015) with the myopic compatibility estimate of
         `propagation.linbp`, whose explicit beliefs are the base predictor's centred soft
         outputs Z - 1/k on non-train nodes and centred one-hot labels on train nodes.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import select_by_val_metric

from .propagation import (
    GRIDS,
    _one_hot_train,
    _to_probs,
    adjacency,
    correct_and_smooth,
    feature_lr,
    linbp,
)

CLP_LAYERS = 50  # paper, §5 "Implementation details" (the notebook uses 20)
CLP_INITS = ("base", "base+train", "train")  # official code's methods 1, 2, 3 (LabelPropagationHeterophilyTog)
SINKHORN_MAX_ITER = 1000  # official `makeDoubleStochastic`
SINKHORN_TOL = 1e-12


def cs_2hop(graph: GraphData, vault: LabelVault, alpha_correct: float = 0.8, alpha_smooth: float = 0.8,
            base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """C&S 2-hop (Lim et al. 2021): `correct_and_smooth` with ``hops=2``."""
    return correct_and_smooth(graph, vault, alpha_correct, alpha_smooth, base=base, hops=2)


def sinkhorn(H: np.ndarray, max_iter: int = SINKHORN_MAX_ITER, tol: float = SINKHORN_TOL) -> tuple[np.ndarray, dict]:
    """Sinkhorn-Knopp as CLP's `utils.makeDoubleStochastic`: divide by the column sums, then by
    the row sums, until the matrix 1-norm (max column sum) of the change is below ``tol``
    (at most ``max_iter`` rounds). Requires strictly positive entries."""
    H = np.array(H, dtype=np.float64)
    for it in range(1, max_iter + 1):
        prev = H
        H = H / H.sum(axis=0, keepdims=True)
        H = H / H.sum(axis=1, keepdims=True)
        if np.linalg.norm(H - prev, 1) < tol:
            return H, {"iterations": it, "converged": True}
    return H, {"iterations": max_iter, "converged": False}


def clp_compatibility(graph: GraphData, vault: LabelVault, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """CLP's `get_myo_h`: B0 = base predictions with train rows replaced by one-hot labels,
    H = S((A Y_train)^T B0) with Y_train one-hot on train rows and 0 elsewhere, A the binary
    adjacency (no self-loops), zero entries set to 1e-7 before Sinkhorn. H[c, c'] sums, over
    train nodes of class c, their neighbours' B0 mass in class c'."""
    Y, idx, _ = _one_hot_train(graph, vault)
    B0 = np.array(base, dtype=np.float64)
    B0[idx] = Y[idx]
    counts = (adjacency(graph) @ Y).T @ B0
    raw = counts.copy()
    counts[counts == 0] = 1e-7
    H, sk = sinkhorn(counts)
    return H, B0, {"H": H.round(4).tolist(), "H_counts": raw.round(4).tolist(), "sinkhorn": sk}


def clp(graph: GraphData, vault: LabelVault, alpha: float = 0.8, init: str = "base+train",
        base: np.ndarray | None = None, layers: int = CLP_LAYERS) -> tuple[np.ndarray, dict]:
    """Compatible Label Propagation (Zhong et al. 2022), official `LabelPropagationHeterophily`
    with edge weights `get_edge_weight` (weight_type 0), no echo cancellation, clamp post-step,
    last layer (no per-layer validation selection):

        w_ij[c] = B0[i, c] * (B0[j] H)[c]        (i receives from neighbour j)
        X <- clamp_[0,1]( alpha * L1rownorm( sum_j w_ij * X[j] ) + (1 - alpha) X0 )

    computed as B0 * (A @ ((B0 H) * X)). ``init`` is the start and restart matrix X0 (the
    code's methods 1-3): "base" = Z (the paper's Eq. 9 restart term (1 - alpha) D_hat),
    "base+train" = B0 (train rows one-hot, as C&S's smoothing step), "train" = one-hot
    train labels, 0 elsewhere (plain LP seeding). Output rows are renormalised to sum to 1
    (all-zero rows -> uniform)."""
    if init not in CLP_INITS:
        raise ValueError(f"clp: init must be one of {CLP_INITS}")
    Z = feature_lr(graph, vault)[0] if base is None else np.asarray(base, dtype=np.float64)
    H, B0, hinfo = clp_compatibility(graph, vault, Z)
    A = adjacency(graph)
    Q = B0 @ H
    if init == "base":
        X0 = Z.copy()
    elif init == "base+train":
        X0 = B0.copy()
    else:
        X0 = _one_hot_train(graph, vault)[0]
    restart = (1.0 - alpha) * X0
    X = X0.copy()
    change = float("nan")
    for _ in range(layers):
        agg = B0 * (A @ (Q * X))
        agg /= np.maximum(np.abs(agg).sum(axis=1, keepdims=True), 1e-12)  # F.normalize(p=1, dim=-1)
        nxt = np.clip(alpha * agg + restart, 0.0, 1.0)
        change = float(np.max(np.abs(nxt - X).sum(axis=1))) if len(X) else 0.0
        X = nxt
    return _to_probs(X), {"alpha": alpha, "init": init, "layers": layers, "last_layer_change_l1": change,
                          "zero_rows": int(np.sum(X.sum(axis=1) <= 0))} | hinfo


def linbp_b(graph: GraphData, vault: LabelVault, scale: float = 0.5, base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """LinBP seeded with the base predictions: `propagation.linbp` (myopic compatibility from
    train-train edges, eps = scale / (rho(W) rho(H_hat))) with explicit beliefs Z - 1/k on
    non-train nodes and one-hot - 1/k on train nodes."""
    Z = feature_lr(graph, vault)[0] if base is None else base
    P, info = linbp(graph, vault, scale=scale, priors=Z)
    return P, info | {"seeded_with_base": True}


# ----------------------------------------------------------- registry + selection
POSTHOC: dict[str, Callable] = {"cs_2hop": cs_2hop, "clp": clp, "linbp_b": linbp_b}
# --posthoc flag value -> record method name
POSTHOC_METHOD = {"cs2hop": "cs_2hop", "clp": "clp", "linbp_b": "linbp_b"}
POSTHOC_GRIDS: dict[str, dict[str, list]] = {
    # 9 points each (rule R6), mirroring the grids of the method each one extends
    "cs_2hop": dict(GRIDS["correct_and_smooth"]),          # alpha_correct x alpha_smooth in {0.5, 0.8, 0.95}
    "clp": {"alpha": [0.1, 0.5, 0.9], "init": list(CLP_INITS)},  # paper range alpha in (0.1, 1) x the code's start matrices
    "linbp_b": dict(GRIDS["linbp"]),                        # the 9 LinBP scales
}
IMPLEMENTATION = {
    "cs_2hop": "propagation.correct_and_smooth(hops=2); Lim et al. 2021, CUAI/Non-Homophily-Large-Scale@82f8f05c",
    "clp": "posthoc.clp; Zhong et al. 2022, zhiqiangzhongddu/TMLR-CLP@2f7901d5",
    "linbp_b": "propagation.linbp(priors=base); Gatterbauer et al. 2015",
}


def select_and_predict_posthoc(name: str, graph: GraphData, vault: LabelVault, metric: str = "accuracy",
                               base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Run post-hoc method ``name`` on base predictions ``base`` (fitted here with feature_lr when
    None) over its 9-point grid and keep the best by the validation value of ``metric``
    (edsp.selection; ties -> first grid point)."""
    fn = POSTHOC[name]
    Z = feature_lr(graph, vault)[0] if base is None else base
    grid = POSTHOC_GRIDS[name]
    keys = list(grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]
    outs = [fn(graph, vault, **c, base=Z) for c in combos]
    best, scores = select_by_val_metric(vault, [o[0] for o in outs], metric)
    probs, info = outs[best]
    return probs, info | {"grid": combos, "selection_metric": metric, "val_scores": scores, "chosen": combos[best],
                          "implementation": IMPLEMENTATION[name]}
