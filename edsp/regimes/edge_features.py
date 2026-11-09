"""Edge features for the H-LEARNED polarity classifier (Master Prompt §2.3c).

Track-B uses node features: [x_i, x_j, |x_i - x_j|, x_i * x_j] on node features
reduced to at most ``max_dim`` dimensions by randomized SVD (documented in
docs/SPEC_AMBIGUITIES.md). Because an undirected edge has no orientation, the
classifier is applied to both orders (i, j) and (j, i) and the two probabilities
are averaged; training uses both orders as well.

Track-A uses label-free structural features only: log degrees (as sum and
absolute difference), common neighbours, Jaccard, Adamic-Adar and the
Hadamard / absolute-difference products of a spectral embedding of the
normalised adjacency.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from edsp.utils.memo import memo
from edsp.utils.numerics import randomized_svd

LARGE_EDGE_COUNT = 5_000_000  # above this, neighbourhood-overlap features are skipped


def reduce_features(X: np.ndarray, max_dim: int = 128, seed: int = 0) -> np.ndarray:
    if X.shape[1] <= max_dim:
        return np.asarray(X, dtype=np.float64)
    X32 = np.asarray(X, dtype=np.float32)
    return np.asarray(randomized_svd(X32 - X32.mean(axis=0), max_dim, seed=seed), dtype=np.float64)


def pair_features(Z: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a, b = Z[src], Z[dst]
    return np.hstack([a, b, np.abs(a - b), a * b])


def _adjacency(n: int, edges: np.ndarray) -> sp.csr_matrix:
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def structural_node_embedding(n: int, edges: np.ndarray, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Label-free; computed once per edge array inside edsp.utils.memo.shared_structure()."""
    def compute() -> np.ndarray:
        A = _adjacency(n, edges)
        deg = np.asarray(A.sum(axis=1)).ravel()
        inv = np.zeros(n)
        np.divide(1.0, np.sqrt(deg), out=inv, where=deg > 0)
        D = sp.diags(inv)
        return randomized_svd(D @ A @ D, dim, seed=seed)

    return memo(f"structural_embedding:{n}:{dim}:{seed}", (edges,), compute)


def batched_edge_scores(
    n: int, edges: np.ndarray, emb: np.ndarray | None, predictors: list, batch: int = 1_000_000
) -> list[np.ndarray]:
    """Scores of every edge for several predictors on graphs above LARGE_EDGE_COUNT, without
    materialising the (m, d) feature matrix (C4: 39.5M edges x 34 float64 = 10.7 GB).

    Features are those of :func:`structural_edge_features` for such graphs (log-degree
    sum / difference and the embedding products; no neighbourhood overlap), computed per
    batch of edges with the same arithmetic; ``predictors`` map a feature block to scores."""
    if len(edges) <= LARGE_EDGE_COUNT:
        raise ValueError("batched_edge_scores is for graphs above LARGE_EDGE_COUNT")
    deg = np.bincount(np.asarray(edges).ravel(), minlength=n).astype(np.float64)
    ld = np.log1p(deg)
    outs = [np.empty(len(edges)) for _ in predictors]
    for start in range(0, len(edges), batch):
        q = edges[start:start + batch]
        u, v = q[:, 0], q[:, 1]
        feats = np.column_stack([ld[u] + ld[v], np.abs(ld[u] - ld[v])])
        if emb is not None:
            feats = np.hstack([feats, emb[u] * emb[v], np.abs(emb[u] - emb[v])])
        for out, f in zip(outs, predictors, strict=True):
            out[start:start + batch] = f(feats)
    return outs


def structural_edge_features(
    n: int, edges: np.ndarray, query: np.ndarray, emb: np.ndarray | None = None, batch: int = 20_000
) -> np.ndarray:
    """Symmetric structural features for the node pairs in ``query`` (q, 2)."""
    A = _adjacency(n, edges)
    deg = np.asarray(A.sum(axis=1)).ravel()
    u, v = query[:, 0], query[:, 1]
    ld = np.log1p(deg)
    cols = [ld[u] + ld[v], np.abs(ld[u] - ld[v])]
    if len(edges) <= LARGE_EDGE_COUNT:
        w = np.zeros(n)
        np.divide(1.0, np.log(deg), out=w, where=deg > 1)
        cn = np.empty(len(u))
        aa = np.empty(len(u))
        for start in range(0, len(u), batch):
            sl = slice(start, start + batch)
            common = A[u[sl]].multiply(A[v[sl]]).tocsr()
            cn[sl] = np.asarray(common.sum(axis=1)).ravel()
            aa[sl] = np.asarray(common @ w).ravel()
        union = deg[u] + deg[v] - cn
        jac = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
        cols += [np.log1p(cn), jac, aa]
    feats = np.column_stack(cols)
    if emb is not None:
        feats = np.hstack([feats, emb[u] * emb[v], np.abs(emb[u] - emb[v])])
    return feats
