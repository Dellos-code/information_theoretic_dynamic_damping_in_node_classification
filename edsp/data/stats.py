"""Dataset statistics (Master Prompt §3.2).

Definitions follow Platonov et al., "Characterizing Graph Datasets for Node
Classification: Homophily-Heterophily Dichotomy and Beyond" (NeurIPS 2023):

* edge homophily      h_edge = |{u,v} in E : y_u = y_v| / |E|
* node homophily      mean over nodes with degree > 0 of the same-class fraction
* adjusted homophily  h_adj = (h_edge - sum_k pbar_k^2) / (1 - sum_k pbar_k^2),
                      pbar_k = D_k / 2|E|, D_k = sum of degrees in class k
* label informativeness (edge LI)
                      LI = I(y_xi; y_eta) / H(y_xi) for a uniformly random
                      directed edge (xi, eta)
                         = 2 - sum p(c1,c2) log p(c1,c2) / sum pbar_c log pbar_c

Full-label ("descriptive") statistics read labels through
``LabelVault.descriptive_labels``; they describe the dataset and are never fed
to a method. The train-label variants only use ``train_labels``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .graph import GraphData


def _directed(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    return src, dst


def class_pair_counts(edges: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """k x k counts of directed edges (each undirected edge counted both ways)."""
    src, dst = _directed(edges)
    mask = (y[src] >= 0) & (y[dst] >= 0)
    return np.bincount(y[src][mask] * k + y[dst][mask], minlength=k * k).reshape(k, k).astype(float)


def edge_homophily(edges: np.ndarray, y: np.ndarray) -> float:
    """Over edges whose endpoints are both labelled (label < 0 = unlabelled)."""
    if len(edges) == 0:
        return float("nan")
    a, b = y[edges[:, 0]], y[edges[:, 1]]
    m = (a >= 0) & (b >= 0)
    return float(np.mean(a[m] == b[m])) if m.any() else float("nan")


def local_homophily(edges: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Per-node same-class fraction among labelled neighbours; NaN for isolated or
    unlabelled nodes (label < 0)."""
    src, dst = _directed(edges)
    lab = (y[src] >= 0) & (y[dst] >= 0)
    src, dst = src[lab], dst[lab]
    deg = np.bincount(src, minlength=n).astype(float)
    same = np.bincount(src, weights=(y[src] == y[dst]).astype(float), minlength=n)
    out = np.full(n, np.nan)
    np.divide(same, deg, out=out, where=deg > 0)
    return out


def local_homophily_bins(g: GraphData, n_bins: int = 5) -> np.ndarray:
    """Descriptive bin (0..n_bins-1, -1 = undefined) of each node's full-label local
    homophily, for the stratified accuracy of Master Prompt §5.7. Uses descriptive
    labels (allowed only in this module); the bins are never inputs to a method."""
    y = g.descriptive_vault().descriptive_labels()
    if y.ndim != 1:
        return np.full(g.num_nodes, -1, dtype=np.int64)
    loc = local_homophily(g.edges, y, g.num_nodes)
    out = np.full(g.num_nodes, -1, dtype=np.int64)
    ok = ~np.isnan(loc)
    out[ok] = np.minimum((loc[ok] * n_bins).astype(np.int64), n_bins - 1)
    return out


def node_homophily(edges: np.ndarray, y: np.ndarray, n: int) -> float:
    return float(np.nanmean(local_homophily(edges, y, n)))


def adjusted_homophily(edges: np.ndarray, y: np.ndarray, k: int) -> float:
    counts = class_pair_counts(edges, y, k)
    total = counts.sum()
    pbar = counts.sum(axis=1) / total
    h = np.trace(counts) / total
    s = float(np.sum(pbar**2))
    return float((h - s) / (1 - s))


def label_informativeness(edges: np.ndarray, y: np.ndarray, k: int) -> float:
    counts = class_pair_counts(edges, y, k)
    p = counts / counts.sum()
    pbar = p.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sum(np.where(p > 0, p * np.log(p), 0.0))
        den = np.sum(np.where(pbar > 0, pbar * np.log(pbar), 0.0))
    return float(2.0 - num / den)


def compatibility_matrix(edges: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    counts = class_pair_counts(edges, y, k)
    rows = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)


def compatibility_distances(H: np.ndarray) -> dict:
    k = H.shape[0]
    uniform_het = (np.ones((k, k)) - np.eye(k)) / (k - 1)
    return {
        "to_identity": float(np.linalg.norm(H - np.eye(k))),
        "to_uniform_heterophily": float(np.linalg.norm(H - uniform_het)),
    }


def structural_stats(g: GraphData) -> dict:
    n, edges = g.num_nodes, g.edges
    deg = np.bincount(edges.ravel(), minlength=n) if len(edges) else np.zeros(n, dtype=int)
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_comp, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp)
    out = {
        "num_nodes": int(n),
        "num_undirected_edges": len(edges),
        "num_classes": int(g.num_classes),
        "components": int(n_comp),
        "largest_component_fraction": float(sizes.max() / n),
        "isolated_nodes": int(np.sum(deg == 0)),
        "degree": {
            "min": int(deg.min()),
            "mean": float(deg.mean()),
            "median": float(np.median(deg)),
            "max": int(deg.max()),
        },
        "num_features": None if g.features is None else int(g.features.shape[1]),
        "preprocessing": g.meta.get("canonicalization", {}),
    }
    if g.edge_sign is not None:
        out["negative_edge_fraction"] = float(np.mean(g.edge_sign < 0))
    if g.edge_attr is not None:
        a = np.asarray(g.edge_attr, dtype=float)
        a = a.reshape(len(a), -1)
        out["edge_attr_quantiles"] = {
            f"col{j}": [float(q) for q in np.quantile(a[:, j], [0, 0.25, 0.5, 0.75, 1])]
            for j in range(a.shape[1])
        }
    return out


def label_stats(g: GraphData, *, train_split: int | None = 0) -> dict:
    """Descriptive full-label statistics plus a train-label-only variant."""
    y = g.descriptive_vault().descriptive_labels()
    if g.multilabel:
        return _multilabel_stats(g, y)
    k, n, edges = g.num_classes, g.num_nodes, g.edges
    counts = np.bincount(y[y >= 0], minlength=k)
    H = compatibility_matrix(edges, y, k)
    loc = local_homophily(edges, y, n)
    hist, _ = np.histogram(loc[~np.isnan(loc)], bins=10, range=(0.0, 1.0))
    out = {
        "class_counts": counts.tolist(),
        "edge_homophily": edge_homophily(edges, y),
        "node_homophily": float(np.nanmean(loc)),
        "adjusted_homophily": adjusted_homophily(edges, y, k),
        "label_informativeness": label_informativeness(edges, y, k),
        "local_homophily_std": float(np.nanstd(loc)),
        "local_homophily_hist_full_labels": hist.tolist(),
        "compatibility_matrix": H.round(6).tolist(),
        "compatibility_distances": compatibility_distances(H),
    }
    if g.edge_sign is not None:
        same = y[edges[:, 0]] == y[edges[:, 1]]
        pos, neg = g.edge_sign > 0, g.edge_sign < 0
        out["signed"] = {
            "positive_edges": int(pos.sum()),
            "negative_edges": int(neg.sum()),
            "zero_sum_edges": int((g.edge_sign == 0).sum()),
            "positive_within_class_fraction": float(same[pos].mean()) if pos.any() else None,
            "negative_between_class_fraction": float((~same[neg]).mean()) if neg.any() else None,
            "sign_label_agreement": float(np.mean(np.where(pos, same, ~same)[pos | neg])) if (pos | neg).any() else None,
        }
    if train_split is not None and g.splits:
        tr_idx, tr_y = g.vault(train_split).train_labels()
        yt = np.full(n, -1, dtype=np.int64)
        yt[tr_idx] = tr_y
        both = (yt[edges[:, 0]] >= 0) & (yt[edges[:, 1]] >= 0)
        e_tr = edges[both]
        out["train_label_view"] = {
            "split_id": g.splits[train_split].split_id,
            "train_train_edges": len(e_tr),
            "edge_homophily": edge_homophily(e_tr, yt) if len(e_tr) else None,
        }
    return out


def _multilabel_stats(g: GraphData, Y: np.ndarray) -> dict:
    edges = g.edges
    per_task = []
    for t in range(Y.shape[1]):
        y = Y[:, t].astype(np.int64)
        per_task.append(
            {
                "positive_rate": float(y.mean()),
                "edge_homophily": edge_homophily(edges, y),
                "adjusted_homophily": adjusted_homophily(edges, y, 2),
                "label_informativeness": label_informativeness(edges, y, 2),
            }
        )
    keys = ["edge_homophily", "adjusted_homophily", "label_informativeness"]
    summary = {
        k: {
            "mean": float(np.nanmean([t[k] for t in per_task])),
            "min": float(np.nanmin([t[k] for t in per_task])),
            "max": float(np.nanmax([t[k] for t in per_task])),
        }
        for k in keys
    }
    return {"num_tasks": int(Y.shape[1]), "per_task_summary": summary, "per_task": per_task}


def dataset_stats(g: GraphData) -> dict:
    return {
        "dataset": g.name,
        "dataset_hash": g.content_hash(),
        "structure": structural_stats(g),
        "labels": label_stats(g),
        "meta": {k: v for k, v in g.meta.items() if k != "canonicalization"},
    }
