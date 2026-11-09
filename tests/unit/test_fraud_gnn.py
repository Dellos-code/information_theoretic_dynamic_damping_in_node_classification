"""B6 fraud GNN baselines (edsp/baselines/fraud_gnn.py): CARE-GNN and PC-GNN.

The neighbour filter and PC-GNN's minority over-sampling are checked against direct ports of
the authors' per-node loops (CARE-GNN ``filter_neighs_ada_threshold``, PC-GNN
``choose_step_neighs``)."""

import importlib.util
import math

import numpy as np
import pytest
from test_regimes import with_test_labels_changed

from edsp.baselines import fraud_gnn as fg
from edsp.data.graph import GraphData, Split

torch_ok = importlib.util.find_spec("torch") is not None
if torch_ok:
    try:
        import torch
    except Exception:  # noqa: BLE001 - broken local installs count as unavailable
        torch_ok = False
needs_torch = pytest.mark.skipif(not torch_ok, reason="torch is unavailable in this environment")


def toy_fraud(seed: int = 0, n: int = 240, frac: float = 0.15) -> GraphData:
    """Three relations (bitmask edge_relation, as the fraud loader): homophilic, mixed, camouflaged."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < frac).astype(np.int64)
    X = np.abs(rng.normal(0, 1, (n, 8)) + y[:, None] * np.r_[1.5, -1.0, np.zeros(6)])
    rows, rel = [], []
    for r, (h, deg) in enumerate([(0.9, 3), (0.3, 10), (0.1, 20)]):
        for i in range(n):
            for _ in range(rng.poisson(deg / 2)):
                pool = np.flatnonzero((y == y[i]) if rng.random() < h else (y != y[i]))
                j = int(rng.choice(pool))
                if i != j:
                    rows.append((min(i, j), max(i, j)))
                    rel.append(1 << r)
    rows, rel = np.array(rows), np.array(rel)
    uk, inv = np.unique(rows[:, 0] * n + rows[:, 1], return_inverse=True)
    mask = np.zeros(len(uk), np.int64)
    np.bitwise_or.at(mask, inv, rel)
    perm = rng.permutation(n)
    split = Split("0", np.sort(perm[: 2 * n // 5]), np.sort(perm[2 * n // 5: 3 * n // 5]), np.sort(perm[3 * n // 5:]), "toy")
    return GraphData("toy-fraud", n, np.stack([uk // n, uk % n], 1), 2, [split], y, features=X.astype(np.float32),
                     edge_relation=mask, meta={"canonicalization": {"relations": ["r0", "r1", "r2"]}})


def test_fraud_grids_have_nine_points():
    assert set(fg.FRAUD_GRIDS) == set(fg.FRAUD_GNNS)
    for grid in fg.FRAUD_GRIDS.values():
        assert np.prod([len(v) for v in grid.values()]) == 9


def test_row_normalize_is_the_authors_normalize():
    X = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
    expect = X / (X.sum(1, keepdims=True) + 0.01)
    assert np.allclose(fg.row_normalize(X), expect)


@needs_torch
def test_relation_csr_has_both_directions_and_self_loops():
    g = toy_fraud()
    csrs, names = fg.relation_csr(g)
    assert names == ["r0", "r1", "r2"] and len(csrs) == 3
    for r, (indptr, indices) in enumerate(csrs):
        e = g.edges[((g.edge_relation >> r) & 1) == 1]
        assert int(indptr[-1]) == 2 * len(e) + g.num_nodes
        nb = {i: set(indices[int(indptr[i]): int(indptr[i + 1])].tolist()) for i in range(g.num_nodes)}
        assert all(i in nb[i] for i in range(g.num_nodes))
        assert all(b in nb[a] and a in nb[b] for a, b in e)


def _authors_filter(center_s0, neigh_ids, neigh_s0, p):
    """CARE-GNN layers.filter_neighs_ada_threshold for one centre (python port)."""
    num_sample = math.ceil(len(neigh_ids) * p)
    diff = np.abs(center_s0 - neigh_s0)
    order = np.argsort(diff, kind="stable")
    if len(neigh_ids) > num_sample + 1:
        return {neigh_ids[i] for i in order[:num_sample]}, num_sample
    return set(neigh_ids), num_sample


@needs_torch
@pytest.mark.parametrize("p", [0.5, 0.5 + 0.02 + 0.02, 0.3, 0.001, 0.999])
def test_vectorised_filter_matches_the_authors_loop(p):
    g = toy_fraud(seed=2)
    csrs, _ = fg.relation_csr(g)
    rng = np.random.default_rng(0)
    s0 = rng.normal(size=g.num_nodes)
    centres = np.sort(rng.choice(g.num_nodes, 60, replace=False))
    for indptr, indices in csrs:
        c, u, d, k = fg.filter_neighbours(torch, torch.as_tensor(centres), torch.as_tensor(s0), indptr, indices, p)
        got = {i: set() for i in range(len(centres))}
        for ci, ui in zip(c.tolist(), u.tolist(), strict=True):
            got[ci].add(ui)
        for i, v in enumerate(centres):
            nb = indices[int(indptr[v]): int(indptr[v + 1])].numpy()
            want, ns = _authors_filter(s0[v], list(nb), s0[nb], p)
            assert got[i] == want and int(k[i]) == ns
        assert np.allclose(d.numpy(), np.abs(s0[centres][c.numpy()] - s0[u.numpy()]))


@needs_torch
def test_pcgnn_oversampling_matches_choose_step_neighs():
    g = toy_fraud(seed=4)
    v = g.vault(0)
    idx, y = v.train_labels()
    train_pos = idx[y == 1]
    csrs, _ = fg.relation_csr(g)
    torch.manual_seed(0)
    model = fg._build("pcgnn", g.features.shape[1], 2, len(csrs))
    x = torch.as_tensor(fg.row_normalize(g.features))
    s0 = model.label_clf(x)[:, 0].detach().numpy()
    centres = idx[:40]
    labels = torch.as_tensor(y[:40])
    rho = 0.5
    captured = []
    orig = fg._mean_aggregate

    def spy(torch_, c, u, b, x_):
        captured.append((c.clone(), u.clone()))
        return orig(torch_, c, u, b, x_)

    fg._mean_aggregate = spy
    try:
        with torch.no_grad():
            model(torch.as_tensor(centres), x, csrs, [0.5] * len(csrs), (labels == 1, torch.as_tensor(train_pos), rho))
    finally:
        fg._mean_aggregate = orig
    for (c, u), (indptr, indices) in zip(captured, csrs, strict=True):
        got = {i: set() for i in range(len(centres))}
        for ci, ui in zip(c.tolist(), u.tolist(), strict=True):
            got[ci].add(ui)
        for i, node in enumerate(centres):
            nb = indices[int(indptr[node]): int(indptr[node + 1])].numpy()
            want, ns = _authors_filter(s0[node], list(nb), s0[nb], 0.5)
            if y[i] == 1:
                k_over = int(ns * rho)
                order = np.argsort(np.abs(s0[node] - s0[train_pos]), kind="stable")
                want |= set(train_pos[order[:k_over]].tolist())
            assert got[i] == want


@needs_torch
@pytest.mark.parametrize("name", fg.FRAUD_GNNS)
def test_each_fraud_gnn_runs_and_ignores_test_labels(name, monkeypatch):
    monkeypatch.setattr(fg, "MAX_EPOCHS", 25)
    g = toy_fraud()
    v = g.vault(0)
    cfg = {k: vals[0] for k, vals in fg.FRAUD_GRIDS[name].items()}
    threads = torch.get_num_threads()
    P, info = fg.train_fraud_gnn(name, g, v, "roc_auc", seed=0, **cfg)
    assert torch.get_num_threads() == threads  # B6 runs single-threaded and restores the setting
    assert P.shape == (g.num_nodes, 2) and np.allclose(P.sum(1), 1, atol=1e-6)
    assert np.isfinite(info["val_score"]) and info["val_score"] > 0.5
    assert v.test_accesses_outside_evaluate() == [] and not any(r.kind == "test" for r in v.access_log)
    if name == "care":
        steps = np.diff(np.array(info["threshold_trace"]), axis=0)
        assert np.allclose(np.abs(steps), fg.CARE_STEP) and len(info["threshold_trace"]) >= 2
    g2 = with_test_labels_changed(g)
    P2, _ = fg.train_fraud_gnn(name, g2, g2.vault(0), "roc_auc", seed=0, **cfg)
    assert np.allclose(P, P2)


@needs_torch
def test_fraud_selection_uses_the_nine_point_grid(monkeypatch):
    monkeypatch.setattr(fg, "MAX_EPOCHS", 5)
    g = toy_fraud()
    P, info = fg.select_and_predict_fraud_gnn("pcgnn", g, g.vault(0))
    assert len(info["grid"]) == len(info["val_scores"]) == 9 and info["chosen"] in info["grid"]
    assert P.shape == (g.num_nodes, 2)
