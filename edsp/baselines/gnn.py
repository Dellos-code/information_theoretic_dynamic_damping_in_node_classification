"""Track-B GNN baselines (Master Prompt B2 / B4 / B6), pure PyTorch (CPU), no PyG.

Models (each with a 9-point grid, rule R6; selection and early stopping on the
validation value of the headline metric through ``edsp.selection``, the only
module allowed to read validation labels):

  mlp      features only (the B0 reference of every GNN)
  gcn      GCN with residual connections + LayerNorm + dropout (the "GCN+" recipe
           of Luo et al., NeurIPS 2024, "Classic GNNs are strong baselines")
  sage     GraphSAGE (mean aggregator, separate root weight), same recipe
  gat      GAT (4 heads × 64), same recipe
  gprgnn   GPR-GNN (Chien et al., ICLR 2021): MLP + K = 10 learnable PPR-initialised hops
  bwgnn    BWGNN (Tang et al., ICML 2022; homogeneous version, order C = 2): Beta-wavelet
           band-pass filters, the fraud-detection reference (B6)

Shared settings: hidden 256, Adam (weight decay 5e-4), at most 500 epochs,
validation every 5 epochs, patience 50 epochs, weights of the best validation
epoch. Grids: layers ∈ {2, 3, 4} × dropout ∈ {0.2, 0.5, 0.7} (mlp, gcn, sage,
gat); alpha ∈ {0.1, 0.5, 0.9} × dropout (gprgnn); lr ∈ {0.01, 0.005, 0.001} ×
dropout (bwgnn). lr = 0.005 elsewhere.
These are faithful re-implementations, not the authors' code; parity against the
published numbers is checked in E3 (data/PUBLISHED_BASELINES.json).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from edsp.data.graph import GraphData
from edsp.data.vault import LabelVault
from edsp.selection import validation_score

DROPOUTS = (0.2, 0.5, 0.7)
GNN_GRIDS = {
    "mlp": {"layers": (2, 3, 4), "dropout": DROPOUTS},
    "gcn": {"layers": (2, 3, 4), "dropout": DROPOUTS},
    "sage": {"layers": (2, 3, 4), "dropout": DROPOUTS},
    "gat": {"layers": (2, 3, 4), "dropout": DROPOUTS},
    "gprgnn": {"alpha": (0.1, 0.5, 0.9), "dropout": DROPOUTS},
    "bwgnn": {"lr": (0.01, 0.005, 0.001), "dropout": DROPOUTS},
}
HIDDEN, MAX_EPOCHS, PATIENCE, EVAL_EVERY, WEIGHT_DECAY = 256, 500, 50, 5, 5e-4


def _torch():
    import torch

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    return torch


def _graph_tensors(graph: GraphData):
    torch = _torch()
    n = graph.num_nodes
    e = graph.edges
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    x = graph.features if graph.features is not None else np.eye(n, dtype=np.float32)
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    ei = torch.as_tensor(np.stack([src, dst]), dtype=torch.long)
    deg = torch.bincount(ei[1], minlength=n).float()
    return x, ei, deg


def _sparse(torch, ei, w, n):
    """Sparse (n x n) operator with out[dst] += w * h[src]."""
    return torch.sparse_coo_tensor(torch.stack([ei[1], ei[0]]), w, (n, n)).coalesce().to_sparse_csr()


def _build(name: str, d_in: int, k: int, cfg: dict, ei, deg, n: int):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional
    drop = cfg["dropout"]
    # normalised weights
    dsl = deg + 1.0
    w_gcn = (dsl[ei[0]] * dsl[ei[1]]).rsqrt()
    w_mean = 1.0 / deg.clamp(min=1.0)[ei[1]]
    w_sym = (deg.clamp(min=1.0)[ei[0]] * deg.clamp(min=1.0)[ei[1]]).rsqrt()
    A_gcn, A_mean, A_sym = (_sparse(torch, ei, w, n) for w in (w_gcn, w_mean, w_sym))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            L = cfg.get("layers", 2)
            self.inp = nn.Linear(d_in, HIDDEN)
            if name in ("mlp", "gcn", "sage", "gat"):
                self.lin = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(L))
                self.norm = nn.ModuleList(nn.LayerNorm(HIDDEN) for _ in range(L))
                if name == "sage":
                    self.root = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(L))
                if name == "gat":
                    self.heads = 4
                    self.att_s = nn.ParameterList(nn.Parameter(torch.randn(self.heads, HIDDEN // 4) * 0.1) for _ in range(L))
                    self.att_d = nn.ParameterList(nn.Parameter(torch.randn(self.heads, HIDDEN // 4) * 0.1) for _ in range(L))
            if name == "gprgnn":
                K, a = 10, cfg["alpha"]
                g = a * (1 - a) ** np.arange(K + 1)
                g[-1] = (1 - a) ** K
                self.gamma = nn.Parameter(torch.tensor(g, dtype=torch.float32))
                self.hid = nn.Linear(HIDDEN, HIDDEN)
            if name == "bwgnn":
                self.hid = nn.Linear(HIDDEN, HIDDEN)
                self.mix = nn.Linear(3 * HIDDEN, HIDDEN)
            self.out = nn.Linear(HIDDEN, k)

        def gat_layer(self, h, i):
            H, dh = self.heads, HIDDEN // self.heads
            hh = h.view(n, H, dh)
            a_s = (hh * self.att_s[i]).sum(-1)
            a_d = (hh * self.att_d[i]).sum(-1)
            src = torch.cat([ei[0], torch.arange(n)])
            dst = torch.cat([ei[1], torch.arange(n)])
            e = F.leaky_relu(a_s[src] + a_d[dst], 0.2)
            m = torch.full((n, H), -1e30).scatter_reduce(0, dst[:, None].expand(-1, H), e, reduce="amax")
            ex = torch.exp(e - m[dst])
            den = torch.zeros(n, H).index_add_(0, dst, ex)
            att = F.dropout(ex / den[dst].clamp(min=1e-16), drop, self.training)
            out = torch.zeros(n, H, dh).index_add_(0, dst, hh[src] * att[..., None])
            return out.reshape(n, HIDDEN)

        def forward(self, x):
            h = F.dropout(F.relu(self.inp(F.dropout(x, drop, self.training))), drop, self.training)
            if name in ("mlp", "gcn", "sage", "gat"):
                for i, (lin, norm) in enumerate(zip(self.lin, self.norm, strict=True)):
                    if name == "mlp":
                        z = lin(h)
                    elif name == "gcn":
                        z = lin(h)
                        z = A_gcn @ z + z / dsl[:, None]
                    elif name == "sage":
                        z = lin(A_mean @ h) + self.root[i](h)
                    else:
                        z = self.gat_layer(lin(h), i)
                    h = h + F.dropout(F.relu(norm(z)), drop, self.training)  # residual + LayerNorm
                return self.out(h)
            if name == "gprgnn":
                z = self.out(F.dropout(F.relu(self.hid(h)), drop, self.training))
                acc, cur = self.gamma[0] * z, z
                for kk in range(1, len(self.gamma)):
                    cur = A_sym @ cur
                    acc = acc + self.gamma[kk] * cur
                return acc
            # bwgnn: Beta wavelets of order C = 2 on L = I - D^-1/2 A D^-1/2
            z = F.relu(self.hid(h))

            def lap(v):
                return v - A_sym @ v

            l1 = lap(z)
            l2 = lap(l1)
            # Beta(1,3)/(2,2)/(3,1) polynomials in L/2, normalised by B(p+1, q+1)
            f0 = 3 * (z - l1 + 0.25 * l2)        # (1 - L/2)^2 / B(1,3)
            f1 = 6 * (0.5 * l1 - 0.25 * l2)      # (L/2)(1 - L/2) / B(2,2)
            f2 = 3 * (0.25 * l2)                 # (L/2)^2 / B(3,1)
            hcat = torch.cat([f0, f1, f2], dim=1)
            return self.out(F.dropout(F.relu(self.mix(hcat)), drop, self.training))

    return Net()


def train_gnn(name: str, graph: GraphData, vault: LabelVault, metric: str = "accuracy", seed: int = 0,
              **cfg) -> tuple[np.ndarray, dict]:
    torch = _torch()
    F = torch.nn.functional
    torch.manual_seed(seed)
    x, ei, deg = _graph_tensors(graph)
    n, k = graph.num_nodes, graph.num_classes
    idx, y = vault.train_labels()
    tidx, ty = torch.as_tensor(idx, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)
    model = _build(name, x.shape[1], k, cfg, ei, deg, n)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.005), weight_decay=WEIGHT_DECAY)
    best, best_state, best_epoch, since = -math.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x)[tidx], ty)
        loss.backward()
        opt.step()
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(x), 1).numpy().astype(np.float64)
            s = validation_score(vault, p, metric)
            s = s if np.isfinite(s) else -math.inf
            if s > best:
                best, best_epoch, since = s, epoch, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                since += EVAL_EVERY
                if since >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).numpy().astype(np.float64)
    return probs, {"best_epoch": best_epoch, "epochs_run": epoch, "val_score": best, "cfg": cfg}


def select_and_predict_gnn(name: str, graph: GraphData, vault: LabelVault, metric: str = "accuracy",
                           seed: int = 0) -> tuple[np.ndarray, dict]:
    grid = GNN_GRIDS[name]
    combos = [dict(zip(grid, vals, strict=True)) for vals in itertools.product(*grid.values())]
    outs = [train_gnn(name, graph, vault, metric, seed=seed, **c) for c in combos]
    scores = [o[1]["val_score"] for o in outs]
    best = int(np.argmax(scores))
    probs, info = outs[best]
    return probs, info | {"grid": combos, "val_scores": scores, "chosen": combos[best], "selection_metric": metric}
