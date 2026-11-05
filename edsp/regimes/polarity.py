"""Edge-polarity regimes h_ij (Master Prompt §2.3).

Every regime uses TRAIN labels only (rule R3). Validation labels are used only
indirectly, through :mod:`edsp.selection`, to choose classifier hyper-parameters.
H-ORACLE lives in :mod:`edsp.regimes.oracle` and is never imported here.

Each function returns a :class:`Polarity`: ``h`` per edge, an optional edge
``keep`` mask (H-TRAINONLY b4 drops edges), an optional learned confidence
(P-LEARNED) and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import select_by_val_edge_auc, select_threshold_by_val_edge_accuracy
from edsp.utils.numerics import fit_logreg

from .edge_features import (
    pair_features,
    reduce_features,
    structural_edge_features,
    structural_node_embedding,
)


@dataclass
class Polarity:
    h: np.ndarray
    keep: np.ndarray | None = None
    confidence: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    score: np.ndarray | None = None  # learned P(heterophilic) per edge used for the decision (H-LEARNED)
    raw_score: np.ndarray | None = None  # class-balanced classifier output before correction (H-LEARNED)


def _train_label_array(vault: LabelVault, n: int) -> np.ndarray:
    idx, y = vault.train_labels()
    if y.ndim != 1:
        raise NotImplementedError("polarity regimes expect single-label data (multi-label: per task)")
    out = np.full(n, -1, dtype=np.int64)
    out[idx] = y
    return out


def h_all(graph: GraphData, value: int) -> Polarity:
    if value not in (0, 1):
        raise ValueError("value must be 0 or 1")
    return Polarity(np.full(len(graph.edges), value, dtype=np.int8), diagnostics={"regime": f"H-ALL{value}"})


def h_native(graph: GraphData) -> Polarity:
    """Exogenous signs: h = 1 iff the (summed) sign is negative (§2.3a)."""
    if graph.edge_sign is None:
        raise ValueError(f"{graph.name}: H-NATIVE needs edge signs")
    s = np.asarray(graph.edge_sign)
    return Polarity((s < 0).astype(np.int8), diagnostics={"regime": "H-NATIVE", "zero_sum_edges": int(np.sum(s == 0))})


def h_trainonly(graph: GraphData, vault: LabelVault, fill: str = "b1") -> Polarity:
    """h = 1[y_i != y_j] on train-train edges; other edges per ``fill``.

    b1: 0 (homophilic), b2: 1 (heterophilic), b3: majority polarity of the
    train-train edges, b4: drop the other edges.
    """
    y = _train_label_array(vault, graph.num_nodes)
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    known = (y[u] >= 0) & (y[v] >= 0)
    het = (y[u] != y[v]) & known
    h = np.zeros(len(u), dtype=np.int8)
    h[het] = 1
    keep = None
    if fill == "b1":
        pass
    elif fill == "b2":
        h[~known] = 1
    elif fill == "b3":
        majority = int(het.sum() > (known.sum() - het.sum())) if known.any() else 0
        h[~known] = majority
    elif fill == "b4":
        keep = known.copy()
    else:
        raise ValueError("fill must be one of b1, b2, b3, b4")
    diag = {
        "regime": f"H-TRAINONLY({fill})",
        "known_edges": int(known.sum()),
        "coverage": float(known.mean()) if len(u) else 0.0,
        "train_edge_heterophily": float(het.sum() / known.sum()) if known.any() else None,
    }
    return Polarity(h, keep, None, diag)


def h_relation(graph: GraphData, vault: LabelVault) -> Polarity:
    """H-RELATION: h_r = 1 iff train-train homophily of relation r < sum_c pi_c^2 (§2.3f)."""
    if graph.edge_relation is None:
        raise ValueError(f"{graph.name}: H-RELATION needs edge relation ids")
    y = _train_label_array(vault, graph.num_nodes)
    pi = np.bincount(y[y >= 0], minlength=graph.num_classes).astype(float)
    pi /= pi.sum()
    baseline = float(np.sum(pi**2))
    rel = np.asarray(graph.edge_relation)
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    known = (y[u] >= 0) & (y[v] >= 0)
    h = np.zeros(len(u), dtype=np.int8)
    per_rel = {}
    for r in np.unique(rel):
        m = (rel == r) & known
        hom = float(np.mean(y[u[m]] == y[v[m]])) if m.any() else float("nan")
        flag = int(hom < baseline) if m.any() else 0
        h[rel == r] = flag
        per_rel[int(r)] = {"train_homophily": hom, "h": flag, "train_edges": int(m.sum())}
    return Polarity(h, diagnostics={"regime": "H-RELATION", "baseline_sum_pi2": baseline, "relations": per_rel})


def h_learned(
    graph: GraphData,
    vault: LabelVault,
    track: str,
    *,
    l2_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    max_dim: int = 64,
    max_train_edges: int = 100_000,
    seed: int = 0,
    decision: str = "prior-corrected",
) -> Polarity:
    """H-LEARNED: class-balanced logistic regression on train-train edges (§2.3c).

    The classifier is trained class-balanced (as §2.3c specifies), which makes its
    output q_raw an estimate of P(het) under EQUAL class priors. ``decision``:

    * ``"prior-corrected"`` (default, PR-4b): restore the train-edge base rate
      pi = P(het | train-train edge) by the Bayes prior-shift correction
      logit q = logit q_raw + log(pi / (1 - pi)); uses TRAIN labels only.
    * ``"val-threshold"``: prior-corrected q, with the decision threshold chosen
      from a fixed grid by accuracy on validation edges (a selection step).
    * ``"balanced"``: the PR-4 behaviour, h = 1[q_raw >= 0.5] (kept as a named
      option; it under-predicts heterophily on strongly heterophilic graphs).

    h = 1[q >= threshold]; the P-LEARNED confidence is max(q, 1 - q). l2 is chosen
    by ROC-AUC on validation edges (invariant to the correction).
    """
    if decision not in ("prior-corrected", "val-threshold", "balanced"):
        raise ValueError("decision must be 'prior-corrected', 'val-threshold' or 'balanced'")
    n, edges = graph.num_nodes, graph.edges
    y = _train_label_array(vault, n)
    u, v = edges[:, 0], edges[:, 1]
    train_mask = (y[u] >= 0) & (y[v] >= 0)
    target = (y[u] != y[v]).astype(np.int64)
    rng = np.random.default_rng(seed)
    train_ids = np.flatnonzero(train_mask)
    if len(train_ids) > max_train_edges:
        train_ids = np.sort(rng.choice(train_ids, max_train_edges, replace=False))
    diag: dict = {"regime": f"H-LEARNED(track {track}, {decision})", "train_edges": int(train_mask.sum()),
                  "train_edges_used": len(train_ids), "decision": decision}
    if len(train_ids) == 0:
        diag["status"] = "not trainable (no train-train edges); fell back to H-ALL0 with p = 0.5"
        return Polarity(np.zeros(len(u), dtype=np.int8), None, np.full(len(u), 0.5), diag)
    if len(np.unique(target[train_ids])) < 2:
        only = int(target[train_ids][0])
        diag["status"] = f"not trainable (all train-train edges have h = {only}); fell back to H-ALL{only} with p = 0.5"
        return Polarity(np.full(len(u), only, dtype=np.int8), None, np.full(len(u), 0.5), diag)

    if track == "B":
        if graph.features is None:
            raise ValueError(f"{graph.name}: Track-B H-LEARNED needs node features")
        Z = reduce_features(graph.features, max_dim, seed)

        def featurize(ids: np.ndarray, flip: bool) -> np.ndarray:
            a, b = (v[ids], u[ids]) if flip else (u[ids], v[ids])
            return pair_features(Z, a, b)

        X_train = np.vstack([featurize(train_ids, False), featurize(train_ids, True)])
        y_train = np.concatenate([target[train_ids], target[train_ids]])
        all_ids = np.arange(len(u))

        def predict(model) -> np.ndarray:
            return 0.5 * (model.predict_proba(featurize(all_ids, False))[:, 1]
                          + model.predict_proba(featurize(all_ids, True))[:, 1])
    elif track == "A":
        emb = structural_node_embedding(n, edges, 16, seed)
        X_all = structural_edge_features(n, edges, edges, emb)
        X_train, y_train = X_all[train_ids], target[train_ids]

        def predict(model) -> np.ndarray:
            return model.predict_proba(X_all)[:, 1]
    else:
        raise ValueError("track must be 'A' or 'B'")

    candidates = [fit_logreg(X_train, y_train, 2, l2=l2, balanced=True) for l2 in l2_grid]
    scores = [predict(m) for m in candidates]
    best, val_aucs = select_by_val_edge_auc(vault, edges, scores)
    q_raw = scores[best]
    pi = float(np.mean(target[train_mask]))
    threshold = 0.5
    if decision == "balanced":
        q = q_raw
    else:
        eps = 1e-12
        qr = np.clip(q_raw, eps, 1 - eps)
        logit = np.log(qr) - np.log1p(-qr) + np.log(pi) - np.log1p(-pi)
        q = 1.0 / (1.0 + np.exp(-logit))
        if decision == "val-threshold":
            grid = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))
            threshold, val_accs = select_threshold_by_val_edge_accuracy(vault, edges, q, grid)
            diag["val_edge_accuracy_by_threshold"] = dict(zip(grid, val_accs, strict=True))
    h = (q >= threshold).astype(np.int8)
    diag.update({"l2_grid": list(l2_grid), "val_edge_auc": val_aucs, "chosen_l2": l2_grid[best],
                 "train_edge_heterophily": pi, "threshold": threshold,
                 "predicted_heterophilic_fraction": float(np.mean(h))})
    return Polarity(h, None, np.maximum(q, 1.0 - q), diag, q, q_raw)
