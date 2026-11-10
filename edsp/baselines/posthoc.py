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

Named variants (PR-39b):
cs_ref       C&S with the reference semantics of Huang et al. 2021 (CUAI/CorrectAndSmooth@b910314,
             `outcome_correlation.double_correlation_autoscale`, train_only=True) and PyG 2.8.1
             `CorrectAndSmooth` / `LabelPropagation`: 50 fixed layers per step, error clamped to
             [-1, 1] after every correct step, autoscale with the factor set to 1 where it is
             infinite or > 1000, smooth start with train rows one-hot, output clamped to [0, 1]
             after every smooth step. `propagation.correct_and_smooth` (records before PR-39b)
             stays unchanged: it iterates to convergence without clamps or cap.
cs_2hop_ref  the same with Lim et al.'s two-hop operator S (S X) (their `correct_smooth.py`, which is
             Huang's code plus ``num_hops``).
clp_es       CLP with the layer (1..50) of every grid point chosen on validation (the official code's
             ``select_eval=True``), selected through edsp.selection, never on test labels.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import select_by_val_metric, validation_score

from .propagation import (
    GRIDS,
    _one_hot_train,
    _sym_norm,
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
CS_REF_LAYERS = 50  # Huang et al.'s OGB configurations, Lim et al.'s main.py / cs_tune_hparams.py, PyG example
AUTOSCALE_CAP = 1000.0  # scale -> 1 where sigma / ||E_i||_1 is infinite or above this (Huang, Lim, PyG)


def cs_2hop(graph: GraphData, vault: LabelVault, alpha_correct: float = 0.8, alpha_smooth: float = 0.8,
            base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """C&S 2-hop (Lim et al. 2021): `correct_and_smooth` with ``hops=2``."""
    return correct_and_smooth(graph, vault, alpha_correct, alpha_smooth, base=base, hops=2)


def cs_reference_raw(apply: Callable[[np.ndarray], np.ndarray], Z: np.ndarray, Y: np.ndarray, idx: np.ndarray,
                     alpha_correct: float, alpha_smooth: float, layers: int = CS_REF_LAYERS) -> tuple[np.ndarray, dict]:
    """Reference C&S on base predictions Z with operator ``apply`` (X -> S X, or S (S X) for 2 hops),
    one-hot train labels Y[idx]; returns the clamped smoothed scores (rows not normalised), as
    PyG `CorrectAndSmooth.correct` + `smooth` (autoscale) and Huang's `double_correlation_autoscale`:

        E0 = Y - Z on train rows, 0 elsewhere;  E <- clamp_[-1,1](a1 * apply(E) + (1 - a1) E0)   (layers x)
        sigma = mean_train ||E0_i||_1;  s_i = sigma / ||E_i||_1, s_i = 1 if infinite / undefined or > 1000
        Zc = Z + s * E (not clipped);  G0 = Zc with train rows Y;  G <- clamp_[0,1](a2 * apply(G) + (1 - a2) G0)

    A row with E_i = 0 keeps Z_i (Huang / Lim replace the 0/0 NaN by the base prediction; PyG would
    propagate the NaN, which only happens when every train node is predicted exactly)."""
    E0 = np.zeros_like(Z)
    E0[idx] = Y[idx] - Z[idx]
    E = E0.copy()
    for _ in range(layers):
        E = np.clip(alpha_correct * apply(E) + (1 - alpha_correct) * E0, -1.0, 1.0)
    sigma = float(np.abs(E0[idx]).sum() / len(idx)) if len(idx) else 0.0
    norm = np.abs(E).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = sigma / norm
    capped = np.isfinite(scale) & (scale > AUTOSCALE_CAP)
    scale[~np.isfinite(scale) | capped] = 1.0
    G0 = Z + scale * E
    G0[idx] = Y[idx]
    G = G0.copy()
    for _ in range(layers):
        G = np.clip(alpha_smooth * apply(G) + (1 - alpha_smooth) * G0, 0.0, 1.0)
    return G, {"sigma": sigma, "autoscale_capped_rows": float(capped.mean()) if len(capped) else 0.0}


def correct_and_smooth_reference(graph: GraphData, vault: LabelVault, alpha_correct: float = 0.8,
                                 alpha_smooth: float = 0.8, base: np.ndarray | None = None, hops: int = 1,
                                 layers: int = CS_REF_LAYERS) -> tuple[np.ndarray, dict]:
    """`cs_reference_raw` with S = D^-1/2 A D^-1/2 (no self-loops; PyG `gcn_norm(add_self_loops=False)`,
    Huang's / Lim's DAD) applied ``hops`` times per layer; rows renormalised to sum to 1."""
    Z = feature_lr(graph, vault)[0] if base is None else np.asarray(base, dtype=np.float64)
    Y, idx, _ = _one_hot_train(graph, vault)
    S = _sym_norm(adjacency(graph))

    def apply(X: np.ndarray) -> np.ndarray:
        for _ in range(hops):
            X = S @ X
        return X

    G, info = cs_reference_raw(apply, Z, Y, idx, alpha_correct, alpha_smooth, layers)
    return _to_probs(G), {"alpha_correct": alpha_correct, "alpha_smooth": alpha_smooth, "hops": hops, "layers": layers,
                          "zero_rows": int(np.sum(G.sum(axis=1) <= 0))} | info


def cs_ref(graph: GraphData, vault: LabelVault, alpha_correct: float = 0.8, alpha_smooth: float = 0.8,
           base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """C&S with the reference semantics (Huang et al. 2021 / PyG), 1 hop."""
    return correct_and_smooth_reference(graph, vault, alpha_correct, alpha_smooth, base=base, hops=1)


def cs_2hop_ref(graph: GraphData, vault: LabelVault, alpha_correct: float = 0.8, alpha_smooth: float = 0.8,
                base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """C&S 2-hop with the reference semantics of Lim et al.'s `correct_smooth.double_correlation_autoscale`."""
    return correct_and_smooth_reference(graph, vault, alpha_correct, alpha_smooth, base=base, hops=2)


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
        base: np.ndarray | None = None, layers: int = CLP_LAYERS, select_layer: bool = False,
        metric: str = "accuracy") -> tuple[np.ndarray, dict]:
    """Compatible Label Propagation (Zhong et al. 2022), official `LabelPropagationHeterophily`
    with edge weights `get_edge_weight` (weight_type 0), no echo cancellation, clamp post-step,
    last layer (no per-layer validation selection):

        w_ij[c] = B0[i, c] * (B0[j] H)[c]        (i receives from neighbour j)
        X <- clamp_[0,1]( alpha * L1rownorm( sum_j w_ij * X[j] ) + (1 - alpha) X0 )

    computed as B0 * (A @ ((B0 H) * X)). ``init`` is the start and restart matrix X0 (the
    code's methods 1-3): "base" = Z (the paper's Eq. 9 restart term (1 - alpha) D_hat),
    "base+train" = B0 (train rows one-hot, as C&S's smoothing step), "train" = one-hot
    train labels, 0 elsewhere (plain LP seeding). Output rows are renormalised to sum to 1
    (all-zero rows -> uniform). ``select_layer`` (variant clp_es; the official ``select_eval=True``)
    returns the layer 1..``layers`` with the best validation ``metric`` (edsp.selection; ties -> the
    earliest layer, as the official strict ``>``; no finite score -> the last layer)."""
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
    best, best_layer, best_X, layer_scores = -np.inf, None, None, []
    for layer in range(1, layers + 1):
        agg = B0 * (A @ (Q * X))
        agg /= np.maximum(np.abs(agg).sum(axis=1, keepdims=True), 1e-12)  # F.normalize(p=1, dim=-1)
        nxt = np.clip(alpha * agg + restart, 0.0, 1.0)
        change = float(np.max(np.abs(nxt - X).sum(axis=1))) if len(X) else 0.0
        X = nxt
        if select_layer:
            score = validation_score(vault, _to_probs(X), metric)
            layer_scores.append(score)
            if np.isfinite(score) and score > best:
                best, best_layer, best_X = score, layer, X
    info = {"alpha": alpha, "init": init, "layers": layers, "last_layer_change_l1": change} | hinfo
    if select_layer:
        if best_X is None:
            best_layer, best_X = layers, X
        X = best_X
        info |= {"selected_layer": best_layer, "layer_val_scores": layer_scores}
    return _to_probs(X), info | {"zero_rows": int(np.sum(X.sum(axis=1) <= 0))}


def clp_es(graph: GraphData, vault: LabelVault, alpha: float = 0.8, init: str = "base+train",
           base: np.ndarray | None = None, metric: str = "accuracy") -> tuple[np.ndarray, dict]:
    """CLP with the layer of each configuration chosen on validation (``select_layer=True``)."""
    return clp(graph, vault, alpha=alpha, init=init, base=base, select_layer=True, metric=metric)


def linbp_b(graph: GraphData, vault: LabelVault, scale: float = 0.5, base: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """LinBP seeded with the base predictions: `propagation.linbp` (myopic compatibility from
    train-train edges, eps = scale / (rho(W) rho(H_hat))) with explicit beliefs Z - 1/k on
    non-train nodes and one-hot - 1/k on train nodes."""
    Z = feature_lr(graph, vault)[0] if base is None else base
    P, info = linbp(graph, vault, scale=scale, priors=Z)
    return P, info | {"seeded_with_base": True}


# ----------------------------------------------------------- registry + selection
POSTHOC: dict[str, Callable] = {"cs_2hop": cs_2hop, "clp": clp, "linbp_b": linbp_b,
                                "cs_ref": cs_ref, "cs_2hop_ref": cs_2hop_ref, "clp_es": clp_es}
# --posthoc flag value -> record method name
POSTHOC_METHOD = {"cs2hop": "cs_2hop", "clp": "clp", "linbp_b": "linbp_b",
                  "csref": "cs_ref", "cs2hopref": "cs_2hop_ref", "clpes": "clp_es"}
POSTHOC_GRIDS: dict[str, dict[str, list]] = {
    # 9 points each (rule R6), mirroring the grids of the method each one extends
    "cs_2hop": dict(GRIDS["correct_and_smooth"]),          # alpha_correct x alpha_smooth in {0.5, 0.8, 0.95}
    "clp": {"alpha": [0.1, 0.5, 0.9], "init": list(CLP_INITS)},  # paper range alpha in (0.1, 1) x the code's start matrices
    "linbp_b": dict(GRIDS["linbp"]),                        # the 9 LinBP scales
    "cs_ref": dict(GRIDS["correct_and_smooth"]),           # the references tune alpha in (0, 1) with Optuna
    "cs_2hop_ref": dict(GRIDS["correct_and_smooth"]),
    "clp_es": {"alpha": [0.1, 0.5, 0.9], "init": list(CLP_INITS)},  # as clp; the layer is chosen inside each point
}
IMPLEMENTATION = {
    "cs_2hop": "propagation.correct_and_smooth(hops=2); Lim et al. 2021, CUAI/Non-Homophily-Large-Scale@82f8f05c",
    "clp": "posthoc.clp; Zhong et al. 2022, zhiqiangzhongddu/TMLR-CLP@2f7901d5",
    "linbp_b": "propagation.linbp(priors=base); Gatterbauer et al. 2015",
    "cs_ref": "posthoc.cs_reference_raw, hops=1; Huang et al. 2021 CUAI/CorrectAndSmooth@b9103146 = PyG 2.8.1 CorrectAndSmooth",
    "cs_2hop_ref": "posthoc.cs_reference_raw, hops=2; Lim et al. 2021, CUAI/Non-Homophily-Large-Scale@82f8f05c correct_smooth.py",
    "clp_es": "posthoc.clp(select_layer=True); Zhong et al. 2022, zhiqiangzhongddu/TMLR-CLP@2f7901d5 select_eval",
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
    extra = {"metric": metric} if name == "clp_es" else {}  # layer selection on the same validation metric
    outs = [fn(graph, vault, **c, base=Z, **extra) for c in combos]
    best, scores = select_by_val_metric(vault, [o[0] for o in outs], metric)
    probs, info = outs[best]
    if name == "clp_es":
        info = info | {"selected_layers": [o[1]["selected_layer"] for o in outs]}
    return probs, info | {"grid": combos, "selection_metric": metric, "val_scores": scores, "chosen": combos[best],
                          "implementation": IMPLEMENTATION[name]}
