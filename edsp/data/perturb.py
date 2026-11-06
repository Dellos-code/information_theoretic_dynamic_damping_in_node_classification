"""PERTURBED-REAL mixing sweep: re-wire the edges of a REAL graph to a controlled
pattern of homophilic / heterophilic mixing (owner request, Nov 2026: "every kind
of mixing, even chaotic").

The nodes, labels, features and splits stay those of the real dataset; only the
edges are re-drawn. Results are PERTURBED-REAL (Master Prompt R4): they are
stored under ``results/perturbed/`` and never enter real-world tables or ranks.

Generative rule (one pass over the canonical undirected edges of the real graph):
for every edge, one endpoint ``a`` (chosen uniformly) is kept as the anchor and its
partner is re-drawn: with probability ``h[a]`` from the anchor's own class,
otherwise from the "heterophilic target" of the anchor's class (all other classes,
or one fixed class pi(y_a) in the ``permutation`` mode). Within the target class
set the partner is drawn with probability proportional to its degree in the real
graph, so the degree distribution is approximately preserved. Duplicates and
self-loops are then merged / removed by ``canonicalize_edges``.

Locality (heterogeneous modes): an edge also lands on its partner, so with fully
global partner draws every node's realised local homophily would drift towards
the graph mean. Nodes are therefore grouped into strata (h rounded to 0.1, the
region in ``regional``), and with probability ``locality`` (parameter ``loc``,
default 0.8; 0 for ``global`` / ``permutation``) the partner is drawn from the
anchor's own stratum, otherwise from the whole target class set.

Modes (``h[a]`` = target homophily of node a):
  global(h)               h[a] = h for every node (homogeneous mixing)
  beta(m, c)              h[a] ~ Beta(m c, (1 - m) c): c < 1 U-shaped (nodes are
                          either homophilic or heterophilic), c > 1 concentrated
  uniform                 h[a] ~ U(0, 1) independently (unstructured random mixing)
  logistic                h[a] = x_t of the logistic map x <- 4x(1 - x) (deterministic
                          chaos; invariant density = arcsine law) along a random node order
  regional(s, hi, lo)     s seed nodes, alternately hi / lo; every node takes the h of
                          its nearest seed in the real graph (spatially coherent regions)
  community(hi, lo)       half of the classes (random) get hi, the other half lo
  permutation(h)          h[a] = h, but heterophilic partners come from one class
                          pi(y_a), pi a random derangement: structured (non-uniform)
                          compatibility, the case the complement operator does not model

The full labels are read here, the only module besides ``edsp.data.stats`` allowed
to call ``descriptive_labels()``: label-dependent re-wiring is the data-generating
process of this sweep, not a method input. Methods still see only train labels.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from edsp.data.graph import GraphData, canonicalize_edges

MODES = ("global", "beta", "uniform", "logistic", "regional", "community", "permutation")


def parse(spec: str) -> tuple[str, dict]:
    """'global:h=0.3' -> ('global', {'h': 0.3}); 'uniform' -> ('uniform', {})."""
    mode, _, rest = spec.partition(":")
    if mode not in MODES:
        raise ValueError(f"unknown mixing mode {mode!r}; known: {MODES}")
    params = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        params[k.strip()] = float(v)
    return mode, params


def _node_homophily_targets(g: GraphData, y: np.ndarray, mode: str, prm: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n, k = g.num_nodes, g.num_classes
    info: dict = {}
    if mode in ("global", "permutation"):
        h = np.full(n, prm.get("h", 0.5))
    elif mode == "beta":
        m, c = prm.get("m", 0.5), prm.get("c", 0.5)
        h = rng.beta(m * c, (1 - m) * c, size=n)
    elif mode == "uniform":
        h = rng.random(n)
    elif mode == "logistic":
        x = rng.uniform(0.01, 0.99)
        seq = np.empty(n)
        for t in range(n + 100):  # 100 burn-in steps
            x = 4.0 * x * (1.0 - x)
            if t >= 100:
                seq[t - 100] = x
        h = np.empty(n)
        h[rng.permutation(n)] = seq
    elif mode == "regional":
        s, hi, lo = int(prm.get("s", 20)), prm.get("hi", 0.9), prm.get("lo", 0.1)
        seeds = rng.choice(n, size=min(s, n), replace=False)
        e = g.edges
        A = sp.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
        _, _, src = dijkstra(A, unweighted=True, indices=seeds, min_only=True, return_predecessors=True)
        seed_h = {int(sd): (hi if i % 2 == 0 else lo) for i, sd in enumerate(seeds)}
        h = np.array([seed_h.get(int(sv), 0.5) for sv in src])  # unreachable nodes (-9999): 0.5
        info["unreachable_nodes"] = int(np.sum(src < 0))
        info["_stratum"] = np.where(src >= 0, src, -1)
    elif mode == "community":
        hi, lo = prm.get("hi", 0.9), prm.get("lo", 0.1)
        homo = rng.permutation(k)[: k // 2]
        hc = np.full(k, lo)
        hc[homo] = hi
        lab = np.where(y >= 0, y, 0)
        h = np.where(y >= 0, hc[lab], 0.5)
        info["homophilic_classes"] = sorted(int(c) for c in homo)
    else:  # pragma: no cover - guarded by parse()
        raise ValueError(mode)
    return np.clip(h, 0.0, 1.0), info


def perturb(g: GraphData, spec: str, seed: int = 0) -> GraphData:
    """Return a PERTURBED-REAL copy of ``g`` whose edges follow mixing ``spec``."""
    mode, prm = parse(spec)
    rng = np.random.default_rng(seed)
    y = g.descriptive_vault().descriptive_labels()
    if y.ndim != 1:
        raise ValueError("mixing sweep needs single-label graphs")
    n, k = g.num_nodes, g.num_classes
    h, info = _node_homophily_targets(g, y, mode, prm, rng)

    deg = np.bincount(g.edges.ravel(), minlength=n).astype(float) + 1.0  # +1: isolated nodes can be drawn
    labelled = y >= 0
    members = [np.flatnonzero(labelled & (y == c)) for c in range(k)]
    if mode == "permutation":
        perm = rng.permutation(k)
        while k > 1 and np.any(perm == np.arange(k)):
            perm = rng.permutation(k)
        info["derangement"] = perm.tolist()

    def sampler(cands: np.ndarray):
        w = deg[cands]
        cdf = np.cumsum(w) / w.sum()
        return lambda size: cands[np.minimum(np.searchsorted(cdf, rng.random(size)), len(cands) - 1)]

    target_other = [members[int(perm[c])] if mode == "permutation" else np.flatnonzero(labelled & (y != c)) for c in range(k)]
    loc = 0.0 if mode in ("global", "permutation") else float(prm.get("loc", 0.8))
    stratum = info.pop("_stratum", None)
    if stratum is None:
        stratum = np.minimum((h * 10).astype(np.int64), 9)
    info["locality"] = loc

    E = g.edges
    flip = rng.random(len(E)) < 0.5
    anchor = np.where(flip, E[:, 1], E[:, 0])
    homo = rng.random(len(E)) < h[anchor]
    local = rng.random(len(E)) < loc
    ya = np.where(labelled[anchor], y[anchor], -1)
    partner = np.empty(len(E), dtype=np.int64)
    anywhere = sampler(np.arange(n))
    cache: dict = {}
    for c in range(-1, k):
        for is_h in (True, False):
            for is_l in (True, False):
                sel = np.flatnonzero((ya == c) & (homo == is_h) & (local == is_l))
                if not len(sel):
                    continue
                if c < 0:
                    partner[sel] = anywhere(len(sel))
                    continue
                base = members[c] if is_h else target_other[c]
                if not is_l:
                    key = (c, is_h, None)
                    if key not in cache:
                        cache[key] = sampler(base) if len(base) else anywhere
                    partner[sel] = cache[key](len(sel))
                    continue
                for st in np.unique(stratum[anchor[sel]]):
                    sub = sel[stratum[anchor[sel]] == st]
                    key = (c, is_h, int(st))
                    if key not in cache:
                        cand = base[stratum[base] == st]
                        cache[key] = sampler(cand) if len(cand) else (sampler(base) if len(base) else anywhere)
                    partner[sub] = cache[key](len(sub))
    raw = np.stack([anchor, partner], axis=1)
    edges, _, _, report = canonicalize_edges(raw, n)
    report |= {"perturbation": {"mode": mode, "params": prm, "seed": seed, "source_dataset": g.name,
                                "source_edges": len(E), "target_mean_h": float(h.mean()), **info}}
    meta = dict(g.meta)
    meta |= {"perturbed_real": True, "canonicalization": report}
    return dataclasses.replace(g, name=label(g.name, spec, seed), edges=edges, edge_attr=None, edge_sign=None,
                               edge_relation=None, meta=meta)


def label(base: str, spec: str, seed: int) -> str:
    return f"{base}~{re.sub(r'[^A-Za-z0-9.=,_-]', '_', spec)}~s{seed}"
