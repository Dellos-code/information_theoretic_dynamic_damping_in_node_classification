"""Node priors v_j and trusts T_j for Track-A and Track-B (Master Prompt §2.1, §2.4).

Track-A (label/structure only): train nodes get one-hot priors with trust T_L
(default infinity), every other node a uniform prior with trust 0.

Track-B (feature prior): a base predictor is trained on train nodes only. PR-4
implements the logistic-regression base predictor (numpy/scipy); the 2-layer MLP
and LightGBM base predictors of §2.1 are listed as not yet implemented in
docs/PROGRESS.md. The l2 strength is chosen by validation accuracy and the
temperature by validation NLL, both through :mod:`edsp.selection`. Trusts:
T_j = tau * (ln k - H(prior_j)) / ln k; with ``train_onehot=True`` train nodes
are overwritten by their one-hot label with T = infinity.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import softmax

from edsp.core import entropy
from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import select_by_val_accuracy, val_nll, val_temperature
from edsp.utils.numerics import fit_logreg

from .edge_features import reduce_features


def track_a_priors(graph: GraphData, vault: LabelVault, T_L: float = math.inf) -> tuple[np.ndarray, np.ndarray, dict]:
    n, k = graph.num_nodes, graph.num_classes
    idx, y = vault.train_labels()
    v = np.full((n, k), 1.0 / k)
    v[idx] = 0.0
    v[idx, y] = 1.0
    trust = np.zeros(n)
    trust[idx] = T_L
    return v, trust, {"track": "A", "T_L": T_L, "labelled_nodes": len(idx)}


def track_b_priors(
    graph: GraphData,
    vault: LabelVault,
    *,
    tau: float = 1.0,
    train_onehot: bool = False,
    l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    max_dim: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if graph.features is None:
        raise ValueError(f"{graph.name}: Track-B needs node features")
    k = graph.num_classes
    X = reduce_features(graph.features, max_dim, seed)
    idx, y = vault.train_labels()
    models = [fit_logreg(X[idx], y, k, l2=l2, balanced=False) for l2 in l2_grid]
    logits = [m.logits(X) for m in models]
    probs = [softmax(lg, axis=1) for lg in logits]
    best, val_accs = select_by_val_accuracy(vault, probs)
    T = val_temperature(vault, logits[best])
    v = models[best].predict_proba(X, temperature=T)
    trust = tau * (math.log(k) - entropy(v)) / math.log(k)
    if train_onehot:
        v[idx] = 0.0
        v[idx, y] = 1.0
        trust[idx] = math.inf
    info = {
        "track": "B", "base_model": "logistic-regression", "l2_grid": list(l2_grid), "val_accuracy": val_accs,
        "chosen_l2": l2_grid[best], "temperature": T, "val_nll_calibrated": val_nll(vault, logits[best], T),
        "tau": tau, "train_onehot": train_onehot, "feature_dim": int(X.shape[1]),
    }
    return v, trust, info
