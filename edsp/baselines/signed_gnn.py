"""B5 signed-GNN baselines (Master Prompt §4 B5): SGCN, SNEA, SDGNN and SiGAT node
embeddings + logistic regression, through PyTorch Geometric Signed Directed (PyGSD).

Library (optional dependency, not in uv.lock; ``requirements/signed-gnn.txt``):
torch-geometric-signed-directed 1.2.0 (PyPI wheel of 2026-09-03, identical to tag 1.2.0 =
commit 2968b55b3378a01ade9f391e194f0077e2c6ee3b of SherylHYX/pytorch_geometric_signed_directed,
MIT licence) on torch_geometric 2.8.1 in its pure-Python mode (no pyg-lib, torch-scatter or
torch-sparse). Model classes used unchanged: ``nn.signed.SGCN`` (Derr et al., ICDM 2018),
``SNEA`` (Li et al., AAAI 2020), ``SDGNN`` (Huang et al., AAAI 2021), ``SiGAT`` (Huang et al.,
ICANN 2019).

Protocol (docs/SPEC_AMBIGUITIES.md, "Signed GNN baselines (B5)"):
  * input: the canonical undirected signed graph; every edge whose summed sign is non-zero
    becomes the two arcs i->j and j->i with that sign; zero-sum pairs carry no sign and are
    dropped (count in the diagnostics). No node features: Track A (structure + train labels).
  * embeddings: each model with its own label-free link-sign objective and the arguments of
    the library's example ``examples/run_link_sign_prediction.py`` (SGCN 2 layers, lambda 5;
    SNEA 2 layers, lambda 4; SDGNN and SiGAT class defaults), spectral (truncated-SVD) initial
    embeddings, Adam lr 0.01, weight decay 1e-3.
  * classifier: multinomial logistic regression on the standardised embeddings, fitted on the
    train labels only (``edsp.utils.numerics.fit_logreg``, as the Track-B base predictor).
  * grid, 9 points (rule R6): embedding size in {20, 32, 64} x logistic-regression l2 in
    {1e-3, 1e-2, 1e-1}; selection on the validation value of the headline metric.
  * early stopping: at most 500 epochs; every 5 epochs the classifier is refitted and scored on
    validation (``edsp.selection.validation_score``); patience 50 epochs; the embedding of the
    best validation epoch is kept. The embedding objective does not depend on l2, so the three
    l2 values share one training run per embedding size, each with its own best epoch and
    patience -- identical to nine separate runs with the same seed.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os
import random
import warnings

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import validation_score
from edsp.utils.numerics import fit_logreg

PYGSD_VERSION = "1.2.0"
PYGSD_COMMIT = "2968b55b3378a01ade9f391e194f0077e2c6ee3b"
PYG_VERSION = "2.8.1"
LIBRARY = f"torch-geometric-signed-directed=={PYGSD_VERSION} (commit {PYGSD_COMMIT[:7]}), torch_geometric=={PYG_VERSION}"

SIGNED_GNNS = ("sgcn", "snea", "sdgnn", "sigat")
DIMS = (20, 32, 64)
LR_L2 = (1e-3, 1e-2, 1e-1)
SIGNED_GRIDS = {m: {"dim": DIMS, "l2": LR_L2} for m in SIGNED_GNNS}
LR, WEIGHT_DECAY = 0.01, 1e-3
MAX_EPOCHS, EVAL_EVERY, PATIENCE = 500, 5, 50
THREADS = int(os.environ.get("EDSP_B56_THREADS", "1"))  # torch intra-op threads (see _single_threaded)


def available() -> bool:
    """True if torch, torch_geometric and PyGSD import (CI installs them only for B5 jobs)."""
    for mod in ("torch", "torch_geometric", "torch_geometric_signed_directed"):
        if importlib.util.find_spec(mod) is None:
            return False
    try:
        _models()
    except Exception:  # noqa: BLE001 - a broken optional install counts as unavailable
        return False
    return True


def _models():
    with warnings.catch_warnings():  # the library emits Syntax/FutureWarnings on import
        warnings.simplefilter("ignore")
        from torch_geometric_signed_directed.nn.signed import SDGNN, SGCN, SNEA, SiGAT
    return {"sgcn": SGCN, "snea": SNEA, "sdgnn": SDGNN, "sigat": SiGAT}


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


def signed_arcs(graph: GraphData) -> tuple[np.ndarray, dict]:
    """(m, 3) int64 rows [i, j, sign] with both directions of every non-zero-sign edge."""
    if graph.edge_sign is None:
        raise ValueError(f"{graph.name}: B5 signed GNNs need edge signs")
    s = np.sign(np.asarray(graph.edge_sign, dtype=np.float64)).astype(np.int64)
    keep = s != 0
    e, s = np.asarray(graph.edges, dtype=np.int64)[keep], s[keep]
    arcs = np.concatenate([np.stack([e[:, 0], e[:, 1], s], 1), np.stack([e[:, 1], e[:, 0], s], 1)])
    info = {"signed_edges": int(keep.sum()), "zero_sign_edges_dropped": int((~keep).sum()),
            "negative_edge_fraction": float(np.mean(s < 0)) if len(s) else None}
    return arcs, info


def _build(name: str, n: int, arcs: np.ndarray, dim: int):
    torch = _torch()
    cls = _models()[name]
    eis = torch.as_tensor(arcs, dtype=torch.long)
    # the spectral initialisation (sklearn TruncatedSVD) needs fewer components than nodes, SDGNN
    # and SiGAT need input size = output size, SGCN and SNEA an even size: on graphs with fewer
    # than dim + 1 nodes (sampson: 18) the size is the largest even number below n
    d = int(min(dim, 2 * ((n - 1) // 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "sgcn":
            model = cls(n, eis, d, d, layer_num=2, lamb=5)
        elif name == "snea":
            model = cls(n, eis, d, d, layer_num=2, lamb=4)
        else:  # sdgnn, sigat: class defaults (as in the library's example script)
            model = cls(n, eis, d, d)
    # SDGNN / SiGAT: a motif with no arc on this graph gives a 1-D empty index tensor, which
    # GATConv rejects; replace it by a (2, 0) edge index (no messages; GATConv's self-loops remain)
    lists = getattr(model, "edge_lists", None)
    if lists is not None:
        for i, ed in enumerate(lists):
            if ed.dim() != 2:
                lists[i] = torch.empty(2, 0, dtype=torch.long)
    return model, d


@_single_threaded
def train_signed_gnn(name: str, graph: GraphData, vault: LabelVault, metric: str = "accuracy", seed: int = 0,
                     dim: int = 32, l2_grid: tuple[float, ...] = LR_L2) -> list[tuple[np.ndarray, dict]]:
    """Train one embedding model; return one (probabilities, info) per l2 in ``l2_grid``."""
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)  # sklearn's TruncatedSVD and the samplers use the global generators
    torch.manual_seed(seed)
    arcs, ainfo = signed_arcs(graph)
    if len(arcs) == 0:
        raise ValueError(f"{graph.name}: no edge with a non-zero sign")
    n, k = graph.num_nodes, graph.num_classes
    idx, y = vault.train_labels()
    model, dim_used = _build(name, n, arcs, dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    L = len(l2_grid)
    best, best_probs = [-math.inf] * L, [None] * L
    best_epoch, since, active = [0] * L, [0] * L, [True] * L
    stop_epoch = [MAX_EPOCHS] * L

    def embed() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            z = model().detach().numpy().astype(np.float64)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def classify(z: np.ndarray, l2: float) -> np.ndarray:
        return fit_logreg(z[idx], y, k, l2=l2, balanced=False, standardize=True).predict_proba(z)

    losses = []
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = model.loss()
        loss.backward()
        opt.step()
        if epoch % EVAL_EVERY:
            continue
        losses.append(float(loss.detach()))
        z = embed()
        for i, l2 in enumerate(l2_grid):
            if not active[i]:
                continue
            p = classify(z, l2)
            s = validation_score(vault, p, metric)
            s = s if np.isfinite(s) else -math.inf
            if s > best[i]:
                best[i], best_probs[i], best_epoch[i], since[i] = s, p, epoch, 0
            else:
                since[i] += EVAL_EVERY
                if since[i] >= PATIENCE:
                    active[i], stop_epoch[i] = False, epoch
        if not any(active):
            break
    z_last = None
    out = []
    for i, l2 in enumerate(l2_grid):
        if best_probs[i] is None:  # no evaluation happened (MAX_EPOCHS < EVAL_EVERY)
            z_last = embed() if z_last is None else z_last
            best_probs[i] = classify(z_last, l2)
        out.append((best_probs[i], {"best_epoch": best_epoch[i], "epochs_run": min(stop_epoch[i], epoch),
                                    "val_score": best[i], "cfg": {"dim": dim, "l2": l2},
                                    "dim_used": dim_used, "loss_every_5_epochs": losses[-3:], **ainfo}))
    return out


def select_and_predict_signed_gnn(name: str, graph: GraphData, vault: LabelVault, metric: str = "accuracy",
                                  seed: int = 0) -> tuple[np.ndarray, dict]:
    grid = SIGNED_GRIDS[name]
    outs = [o for dim in grid["dim"] for o in train_signed_gnn(name, graph, vault, metric, seed, dim, grid["l2"])]
    combos = [o[1]["cfg"] for o in outs]
    scores = [o[1]["val_score"] for o in outs]
    best = int(np.argmax(scores))
    probs, info = outs[best]
    return probs, info | {"grid": combos, "val_scores": scores, "chosen": combos[best], "selection_metric": metric,
                          "best_epochs": [o[1]["best_epoch"] for o in outs], "library": LIBRARY}
