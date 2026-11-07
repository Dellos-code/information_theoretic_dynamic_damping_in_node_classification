"""Train-only compatibility estimate for the named variant EDSP-fullH (Master Prompt
§2.5: "C replaced by row-normalised compatibility Ĥ estimated from train–train edges,
Laplace-smoothed").

Ĥ is estimated on the train–train edges that the polarity regime marks heterophilic
(h_ij = 1), because the operator replaces C only on those edges: Ĥ[a, b] estimates
P(class of the receiving end = b | class of the sending end = a, edge heterophilic).
Counts are symmetric (each undirected edge counts in both directions) plus Laplace
smoothing ``alpha`` on every cell. If no train–train edge is heterophilic, the
canonical complement (J − I)/(k − 1) is returned and ``fallback`` is set.
Only train labels are read (vault.train_labels(), rule R3).
"""

from __future__ import annotations

import numpy as np

from edsp.data.vault import LabelVault


def estimate_heterophilic_compatibility(edges: np.ndarray, heterophily: np.ndarray, vault: LabelVault, k: int,
                                        alpha: float = 1.0) -> tuple[np.ndarray, dict]:
    idx, y = vault.train_labels()
    lab = np.full(int(max(edges.max(initial=-1) + 1, idx.max(initial=-1) + 1)), -1, dtype=np.int64)
    lab[idx] = y
    e = np.asarray(edges, dtype=np.int64)
    a, b = lab[e[:, 0]], lab[e[:, 1]]
    use = (a >= 0) & (b >= 0) & (np.asarray(heterophily) == 1)
    counts = np.zeros((k, k))
    np.add.at(counts, (a[use], b[use]), 1.0)
    np.add.at(counts, (b[use], a[use]), 1.0)
    info = {"train_train_heterophilic_edges": int(use.sum()), "laplace_alpha": alpha}
    if not use.any():
        return (np.ones((k, k)) - np.eye(k)) / (k - 1), info | {"fallback": "complement"}
    H = counts + alpha
    H /= H.sum(axis=1, keepdims=True)
    comp = (np.ones((k, k)) - np.eye(k)) / (k - 1)
    return H, info | {"fallback": None, "distance_to_complement": float(np.linalg.norm(H - comp)),
                      "diagonal_mass": float(np.trace(H) / k)}
