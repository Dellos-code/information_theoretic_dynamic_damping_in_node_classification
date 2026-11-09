"""B6 fraud-detection GNN baselines (Master Prompt §4 B6): CARE-GNN and PC-GNN, re-implemented
in plain PyTorch (CPU, no PyG / DGL) from the authors' code, vectorised over mini-batches.

  care   CARE-GNN (Dou et al., CIKM 2020), YingtongDou/CARE-GNN at commit
         a64ff7523e187a24251f7ca88435d2c9d8f7dcd9 (Apache-2.0), files ``layers.py``
         (InterAgg, IntraAgg, RLModule, filter_neighs_ada_threshold, threshold_inter_agg),
         ``model.py`` (OneLayerCARE), ``train.py``, ``utils.py`` (normalize, undersample).
  pcgnn  PC-GNN (Liu et al., WWW 2021), PonderLY/PC-GNN at commit
         9d7d7fae491081178b6e12e193fb89cfc330f2be (no licence file; re-implemented, no code
         copied), files ``src/layers.py`` (InterAgg, IntraAgg, choose_step_neighs,
         choose_step_test), ``src/model.py`` (PCALayer), ``src/utils.py`` (pick_step),
         ``src/model_handler.py``, ``config/pcgnn_{amazon,yelpchi}.yml``.

What is reproduced (both: one layer, relation graphs with self-loops as in the authors'
``sparse_to_adjlist``, node features row-normalised by ``row sum + 0.01`` as in their
``normalize``, a linear label-aware classifier whose class-0 logit defines the neighbour
distance |s(v) - s(u)|, top-ceil(p_r * deg) neighbours per relation r unless deg <= that
number + 1, mean intra-relation aggregation, loss CE(GNN) + lambda * CE(label-aware)):
  care   per epoch all train frauds + an equal random sample of train benign nodes; inter-
         relation aggregation ReLU(x W + sum_r p_r * ReLU(mean_r) W); p_r start at 0.5 and are
         moved by +-0.02 per epoch from the third epoch on by the RL rule of ``RLModule`` (the
         mean selected-neighbour distance of the batch frauds decreased -> +0.02, else -0.02;
         clipped to [0.001, 0.999]); softmax output. Adam, weight decay 1e-3, embedding 64.
  pcgnn  "pick": 2 x #train-frauds training nodes drawn with replacement with probability
         proportional to (homogeneous degree + 1) / #train nodes of the node's class; "choose":
         fixed p_r = 0.5 under-sampling of neighbours and, for fraud centres during training,
         int(rho * ceil(p_r deg)) extra neighbours = the nearest train frauds by the same
         distance; intra ReLU([x || mean_r] W_r), inter ReLU([x || h_1 || ... || h_R] W);
         score of the fraud class = sigmoid(logit_1) as in the authors' ``to_prob`` (we report
         [1 - sigmoid(logit_1), sigmoid(logit_1)]). Adam, weight decay 5e-4 (amazon-fraud) or
         1e-3 (yelpchi), alpha = 2, embedding 64.
Batch size 256 (amazon-fraud) / 1024 (yelpchi), as in both repositories.

Protocol of this repository (PREREGISTRATION A7 / A9, docs/SPEC_AMBIGUITIES.md "Fraud GNN
baselines (B6)"): Track B (node features), at most 500 epochs, validation ROC-AUC (headline
metric, through ``edsp.selection.validation_score``) every 5 epochs, patience 50 epochs, the
weights (and CARE thresholds) of the best validation epoch; 9-point grids (rule R6):
  care   lr in {0.01, 0.005, 0.001} x lambda_1 (similarity-loss weight) in {1, 2, 4}
  pcgnn  lr in {0.01, 0.005, 0.001} x rho (minority over-sampling ratio) in {0.2, 0.5, 0.8}
Early-stopping evaluations score the validation nodes only (their indices, never their labels,
are read here); the final prediction covers every node.
"""

from __future__ import annotations

import copy
import functools
import itertools
import math
import os

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import validation_score

CARE_COMMIT = "a64ff7523e187a24251f7ca88435d2c9d8f7dcd9"
PCGNN_COMMIT = "9d7d7fae491081178b6e12e193fb89cfc330f2be"
FRAUD_GNNS = ("care", "pcgnn")
FRAUD_GRIDS = {
    "care": {"lr": (0.01, 0.005, 0.001), "lambda_1": (1.0, 2.0, 4.0)},
    "pcgnn": {"lr": (0.01, 0.005, 0.001), "rho": (0.2, 0.5, 0.8)},
}
EMB = 64
MAX_EPOCHS, EVAL_EVERY, PATIENCE = 500, 5, 50
THREADS = int(os.environ.get("EDSP_B56_THREADS", "1"))  # torch intra-op threads (see _single_threaded)
BATCH_SIZE = {"yelpchi": 1024, "amazon-fraud": 256}
DEFAULT_BATCH = 256
CARE_WEIGHT_DECAY, CARE_STEP, THRESHOLD0 = 1e-3, 0.02, 0.5
PCGNN_WEIGHT_DECAY = {"yelpchi": 1e-3, "amazon-fraud": 5e-4}
PCGNN_ALPHA = 2.0
EVAL_CHUNK = 2048  # centres per forward pass at evaluation time (memory bound)


def _torch():
    import torch

    return torch


def _single_threaded(fn):
    """Run ``fn`` with THREADS torch threads and restore the previous setting afterwards, so
    that the B2/B4 GNNs of the same process keep theirs. Default 1 (env EDSP_B56_THREADS):
    these models are dominated by many small operations and Python-side sampling, where
    intra-op threads only add overhead; it is also the setting of the PR-29 timing estimates
    and makes runs independent of the runner's core count."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch = _torch()
        old = torch.get_num_threads()
        torch.set_num_threads(THREADS)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(old)
    return wrapper


def _base_name(graph: GraphData) -> str:
    return graph.name.split("~")[0]


def row_normalize(X: np.ndarray) -> np.ndarray:
    """CARE-GNN / PC-GNN ``utils.normalize``: rows divided by (row sum + 0.01)."""
    X = np.asarray(X, dtype=np.float64)
    with np.errstate(divide="ignore"):
        inv = 1.0 / (X.sum(axis=1) + 0.01)
    inv[~np.isfinite(inv)] = 0.0
    return (X * inv[:, None]).astype(np.float32)


def relation_csr(graph: GraphData) -> tuple[list, list[str]]:
    """Per relation r: CSR (indptr, indices) of the edges whose bitmask has bit r, both
    directions plus one self-loop per node; neighbours sorted by id."""
    torch = _torch()
    if graph.edge_relation is None:
        raise ValueError(f"{graph.name}: B6 fraud GNNs need relation-typed edges")
    rel = np.asarray(graph.edge_relation, dtype=np.int64)
    R = max(1, int(rel.max()).bit_length())
    names = list(graph.meta.get("canonicalization", {}).get("relations") or [f"r{r}" for r in range(R)])[:R]
    n = graph.num_nodes
    loops = np.arange(n, dtype=np.int64)
    out = []
    for r in range(R):
        e = np.asarray(graph.edges, dtype=np.int64)[((rel >> r) & 1) == 1]
        src = np.concatenate([e[:, 0], e[:, 1], loops])
        dst = np.concatenate([e[:, 1], e[:, 0], loops])
        order = np.lexsort((dst, src))
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
        out.append((torch.as_tensor(indptr), torch.as_tensor(dst[order])))
    return out, names


def filter_neighbours(torch, centres, s0, indptr, indices, p: float):
    """Vectorised ``filter_neighs_ada_threshold`` / ``choose_step_test``: for every centre keep
    the ceil(p * deg) neighbours (self-loop included) nearest by |s0(centre) - s0(u)|, or all
    of them if deg <= that number + 1. Returns (centre position, neighbour, distance) of the
    kept arcs, ordered by centre then neighbour id, and the per-centre sample size."""
    starts = indptr[centres]
    deg = indptr[centres + 1] - starts
    b = len(centres)
    c = torch.repeat_interleave(torch.arange(b), deg)
    first = torch.cumsum(deg, 0) - deg
    off = torch.arange(int(deg.sum())) - first[c]
    u = indices[starts[c] + off]
    d = (s0[centres][c] - s0[u]).abs()
    k = torch.ceil(deg.double() * p).long()  # Python's math.ceil(len(neighs) * threshold)
    keep_all = deg <= k + 1
    order = torch.argsort(d, stable=True)
    order = order[torch.argsort(c[order], stable=True)]
    rank = torch.empty_like(off)
    rank[order] = off  # centre blocks keep their positions, so the rank inside a block is `off`
    keep = keep_all[c] | (rank < k[c])
    return c[keep], u[keep], d[keep], k


def _mean_aggregate(torch, c, u, b: int, x):
    cnt = torch.bincount(c, minlength=b)
    crow = torch.zeros(b + 1, dtype=torch.long)
    crow[1:] = torch.cumsum(cnt, 0)
    vals = (1.0 / cnt.clamp(min=1).float())[c]
    A = torch.sparse_csr_tensor(crow, u, vals, (b, x.shape[0]), check_invariants=False)
    return A @ x


def _build(name: str, d: int, k: int, R: int):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    def param(*shape):
        w = nn.Parameter(torch.empty(*shape))
        nn.init.xavier_uniform_(w)
        return w

    class Care(nn.Module):
        """OneLayerCARE with the 'GNN' (threshold-weighted) inter-relation aggregator."""

        def __init__(self) -> None:
            super().__init__()
            self.label_clf = nn.Linear(d, k)
            self.weight = param(d, EMB)
            self.out = param(EMB, k)

        def forward(self, centres, x, csrs, thresholds, train=None):
            s = self.label_clf(x)
            s0 = s[:, 0].detach()
            h = x[centres] @ self.weight
            rl = []
            for r, (indptr, indices) in enumerate(csrs):
                c, u, dist, _ = filter_neighbours(torch, centres, s0, indptr, indices, thresholds[r])
                agg = F.relu(_mean_aggregate(torch, c, u, len(centres), x))
                h = h + (agg @ self.weight) * thresholds[r]
                rl.append((c, dist))
            return F.relu(h) @ self.out, s[centres], rl

    class PCGNN(nn.Module):
        """PCALayer: pick (sampler, outside), choose (neighbour sampler), aggregate."""

        def __init__(self) -> None:
            super().__init__()
            self.label_clf = nn.Linear(d, k)
            self.intra = nn.ParameterList(param(2 * d, EMB) for _ in range(R))
            self.weight = param(EMB * R + d, EMB)
            self.out = param(EMB, k)

        def forward(self, centres, x, csrs, thresholds, train=None):
            s = self.label_clf(x)
            s0 = s[:, 0].detach()
            xc = x[centres]
            parts = [xc]
            n = x.shape[0]
            for r, (indptr, indices) in enumerate(csrs):
                c, u, _, ks = filter_neighbours(torch, centres, s0, indptr, indices, thresholds[r])
                if train is not None:  # choose_step_neighs: over-sample train frauds for fraud centres
                    is_pos, train_pos, rho = train
                    pc = torch.nonzero(is_pos).flatten()
                    n_over = (ks[pc].double() * rho).long()  # int(num_sample * rho)
                    kk = int(min(int(n_over.max()) if len(pc) else 0, len(train_pos)))
                    if kk > 0:
                        D = (s0[centres[pc]][:, None] - s0[train_pos][None, :]).abs()
                        _, nearest = torch.topk(D, kk, dim=1, largest=False, sorted=True)
                        m = torch.arange(kk)[None, :] < n_over[:, None]
                        c = torch.cat([c, pc[:, None].expand(-1, kk)[m]])
                        u = torch.cat([u, train_pos[nearest][m]])
                        key = torch.unique(c * n + u)  # neighbour *sets*, sorted by centre then id
                        c, u = key // n, key % n
                agg = _mean_aggregate(torch, c, u, len(centres), x)
                parts.append(F.relu(torch.cat([xc, agg], 1) @ self.intra[r]))
            h = F.relu(torch.cat(parts, 1) @ self.weight)
            return h @ self.out, s[centres], None

    return Care() if name == "care" else PCGNN()


def _fraud_scores(torch, name: str, logits):
    """(b, k) probabilities: softmax (CARE-GNN ``to_prob``); PC-GNN's ``to_prob`` scores the
    fraud class by sigmoid(logit_1), reported as [1 - sigmoid, sigmoid] (binary graphs)."""
    if name == "pcgnn" and logits.shape[1] == 2:
        p1 = torch.sigmoid(logits[:, 1])
        return torch.stack([1 - p1, p1], 1)
    return torch.softmax(logits, 1)


def _predict(torch, name, model, nodes, x, csrs, thresholds, n: int, k: int) -> np.ndarray:
    P = np.full((n, k), 1.0 / k)
    model.eval()
    with torch.no_grad():
        for a in range(0, len(nodes), EVAL_CHUNK):
            ch = nodes[a: a + EVAL_CHUNK]
            logits, _, _ = model(ch, x, csrs, thresholds)
            P[ch.numpy()] = _fraud_scores(torch, name, logits).numpy().astype(np.float64)
    return P


@_single_threaded
def train_fraud_gnn(name: str, graph: GraphData, vault: LabelVault, metric: str = "roc_auc", seed: int = 0,
                    **cfg) -> tuple[np.ndarray, dict]:
    torch = _torch()
    F = torch.nn.functional
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if graph.features is None:
        raise ValueError(f"{graph.name}: B6 fraud GNNs are Track B and need node features")
    n, k = graph.num_nodes, graph.num_classes
    x = torch.as_tensor(row_normalize(graph.features))
    csrs, rel_names = relation_csr(graph)
    R = len(csrs)
    base = _base_name(graph)
    bs = BATCH_SIZE.get(base, DEFAULT_BATCH)
    idx, y = vault.train_labels()
    pos, neg = idx[y == 1], idx[y != 1]  # class 1 = fraud
    if len(pos) == 0:
        raise ValueError(f"{graph.name}: no fraud node among the train labels")
    y_full = np.full(n, -1, dtype=np.int64)
    y_full[idx] = y  # train labels only
    model = _build(name, x.shape[1], k, R)
    if name == "care":
        wd, lam = CARE_WEIGHT_DECAY, cfg["lambda_1"]
    else:
        wd, lam = PCGNN_WEIGHT_DECAY.get(base, 1e-3), PCGNN_ALPHA
        homo_deg = np.bincount(np.asarray(graph.edges).ravel(), minlength=n) + 1  # union graph + self-loop
        lf = np.where(y == 1, len(pos), len(idx)).astype(np.float64)  # (sum(y) - len(y)) * y + len(y)
        pick_p = homo_deg[idx] / lf
        pick_p = pick_p / pick_p.sum()
        train_pos = torch.as_tensor(pos, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    thresholds = [THRESHOLD0] * R
    score_log: list[list[float]] = []  # CARE: per-batch mean selected distance of the batch frauds
    thr_trace = [list(thresholds)]
    val_nodes = torch.as_tensor(np.asarray(vault.val_idx), dtype=torch.long)  # indices only
    best, best_state, best_thr, best_epoch, since = -math.inf, None, list(thresholds), 0, 0
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        if name == "care":  # undersample(): all train frauds + as many random benign nodes
            sample = np.concatenate([pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)])
        else:  # pick_step(): label-balanced, degree-weighted, with replacement
            sample = rng.choice(idx, size=2 * len(pos), replace=True, p=pick_p)
        rng.shuffle(sample)
        batches = [sample[a: a + bs] for a in range(0, len(sample), bs)]
        for nodes in batches:
            centres = torch.as_tensor(nodes, dtype=torch.long)
            labels = torch.as_tensor(y_full[nodes], dtype=torch.long)
            opt.zero_grad()
            train = None if name == "care" else (labels == 1, train_pos, cfg["rho"])
            logits, label_scores, rl = model(centres, x, csrs, thresholds, train)
            loss = F.cross_entropy(logits, labels) + lam * F.cross_entropy(label_scores, labels)
            loss.backward()
            opt.step()
            if name == "care":  # RLModule, called once per batch after the forward pass
                nb = len(batches)
                if len(score_log) % nb == 0 and len(score_log) >= 2 * nb:
                    prev = np.nanmean(np.array(score_log[-2 * nb: -nb]), axis=0)
                    cur = np.nanmean(np.array(score_log[-nb:]), axis=0)
                    thresholds = [t + CARE_STEP if p_ - c_ >= 0 else t - CARE_STEP
                                  for t, p_, c_ in zip(thresholds, prev, cur, strict=True)]
                    thresholds = [0.999 if t > 1 else (0.001 if t < 0 else t) for t in thresholds]
                    thr_trace.append(list(thresholds))
                is_pos = labels == 1
                row = []
                for c, dist in rl:
                    sel = is_pos[c]
                    row.append(float(dist[sel].sum() / sel.sum()) if bool(sel.any()) else float("nan"))
                score_log.append(row)
        if epoch % EVAL_EVERY == 0:
            P = _predict(torch, name, model, val_nodes, x, csrs, thresholds, n, k)
            s = validation_score(vault, P, metric)
            s = s if np.isfinite(s) else -math.inf
            if s > best:
                best, best_epoch, since = s, epoch, 0
                best_state, best_thr = copy.deepcopy(model.state_dict()), list(thresholds)
            else:
                since += EVAL_EVERY
                if since >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
        thresholds = best_thr
    probs = _predict(torch, name, model, torch.arange(n), x, csrs, thresholds, n, k)
    info = {"best_epoch": best_epoch, "epochs_run": epoch, "val_score": best, "cfg": cfg,
            "thresholds": [round(t, 6) for t in thresholds], "relations": rel_names, "batch_size": bs,
            "weight_decay": wd, "lambda": lam, "embedding": EMB}
    if name == "care":
        info["threshold_trace"] = [[round(t, 3) for t in tr] for tr in thr_trace]  # start + one entry per RL update
    return probs, info


def select_and_predict_fraud_gnn(name: str, graph: GraphData, vault: LabelVault, metric: str = "roc_auc",
                                 seed: int = 0) -> tuple[np.ndarray, dict]:
    grid = FRAUD_GRIDS[name]
    combos = [dict(zip(grid, vals, strict=True)) for vals in itertools.product(*grid.values())]
    outs = [train_fraud_gnn(name, graph, vault, metric, seed=seed, **c) for c in combos]
    scores = [o[1]["val_score"] for o in outs]
    best = int(np.argmax(scores))
    probs, info = outs[best]
    return probs, info | {"grid": combos, "val_scores": scores, "chosen": combos[best], "selection_metric": metric,
                          "best_epochs": [o[1]["best_epoch"] for o in outs],
                          "source": {"care": f"YingtongDou/CARE-GNN@{CARE_COMMIT[:7]}",
                                     "pcgnn": f"PonderLY/PC-GNN@{PCGNN_COMMIT[:7]}"}[name] + " (re-implementation)"}
