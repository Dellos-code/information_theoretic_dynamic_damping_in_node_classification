"""B5 signed-GNN baselines (edsp/baselines/signed_gnn.py; PyTorch Geometric Signed Directed).

The model tests skip when torch or the optional library is missing (it is not in uv.lock;
CI installs requirements/signed-gnn.txt in a separate job and sets EDSP_REQUIRE_PYGSD=1 so
that a broken install fails instead of skipping)."""

import os

import numpy as np
import pytest
from test_regimes import toy_graph, with_test_labels_changed

from edsp.baselines import signed_gnn as sg
from edsp.data.graph import GraphData, Split

pygsd_ok = sg.available()
if not pygsd_ok and os.getenv("EDSP_REQUIRE_PYGSD") == "1":
    raise RuntimeError("EDSP_REQUIRE_PYGSD=1 but torch / torch_geometric / torch-geometric-signed-directed do not import")
needs_pygsd = pytest.mark.skipif(not pygsd_ok, reason="torch-geometric-signed-directed is not installed (optional)")


def test_signed_grids_have_nine_points():
    assert set(sg.SIGNED_GRIDS) == set(sg.SIGNED_GNNS)
    for grid in sg.SIGNED_GRIDS.values():
        assert np.prod([len(v) for v in grid.values()]) == 9


def test_signed_arcs_both_directions_and_zero_sum_pairs_dropped():
    g = toy_graph(signed=True)
    sign = g.edge_sign.copy()
    sign[:5] = 0.0  # reciprocal votes that cancel: no sign
    g.edge_sign = sign
    arcs, info = sg.signed_arcs(g)
    m = len(g.edges) - 5
    assert arcs.shape == (2 * m, 3) and info["zero_sign_edges_dropped"] == 5 and info["signed_edges"] == m
    assert set(np.unique(arcs[:, 2])) <= {-1, 1}
    fwd = {(int(a), int(b), int(s)) for a, b, s in arcs}
    assert all((b, a, s) in fwd for a, b, s in fwd)
    with pytest.raises(ValueError):
        sg.signed_arcs(toy_graph())


@needs_pygsd
@pytest.mark.parametrize("name", sg.SIGNED_GNNS)
def test_each_signed_gnn_learns_the_toy_graph_and_ignores_test_labels(name, monkeypatch):
    monkeypatch.setattr(sg, "MAX_EPOCHS", 30)
    g = toy_graph(signed=True)
    v = g.vault(0)
    outs = sg.train_signed_gnn(name, g, v, seed=0, dim=20)
    assert len(outs) == len(sg.LR_L2)
    idx, y = v.train_labels()
    for P, info in outs:
        assert P.shape == (g.num_nodes, g.num_classes) and np.allclose(P.sum(1), 1, atol=1e-6)
        assert np.isfinite(info["val_score"]) and info["best_epoch"] % sg.EVAL_EVERY == 0
    train_acc = float(np.mean(outs[0][0][idx].argmax(1) == y))
    assert train_acc > 0.5, (name, train_acc)
    assert v.test_accesses_outside_evaluate() == []
    assert not any(r.kind == "test" for r in v.access_log)
    g2 = with_test_labels_changed(g)
    outs2 = sg.train_signed_gnn(name, g2, g2.vault(0), seed=0, dim=20)
    for (P, _), (P2, _) in zip(outs, outs2, strict=True):
        assert np.allclose(P, P2)


@needs_pygsd
@pytest.mark.parametrize("name", ["sdgnn", "sigat"])
def test_tiny_graph_caps_spectral_dimension_and_handles_empty_motifs(name, monkeypatch):
    """Sampson has 18 nodes: the TSVD initialisation needs < n components, and on such a graph
    several SDGNN / SiGAT motif lists are empty."""
    monkeypatch.setattr(sg, "MAX_EPOCHS", 10)
    rng = np.random.default_rng(3)
    n = 18
    y = np.repeat([0, 1, 2], 6)
    e = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2])
    s = np.where(y[e[:, 0]] == y[e[:, 1]], 1.0, -1.0)
    split = Split("0", np.arange(0, n, 2), np.arange(1, n, 4), np.arange(3, n, 4), "toy")
    g = GraphData("tiny", n, e, 3, [split], y, edge_sign=s)
    outs = sg.train_signed_gnn(name, g, g.vault(0), seed=0, dim=64)
    assert outs[0][1]["dim_used"] == 16  # largest even size below n = 18
    assert outs[0][0].shape == (n, 3)


@needs_pygsd
def test_selection_uses_the_nine_point_grid(monkeypatch):
    monkeypatch.setattr(sg, "MAX_EPOCHS", 10)
    g = toy_graph(signed=True)
    P, info = sg.select_and_predict_signed_gnn("sgcn", g, g.vault(0))
    assert len(info["grid"]) == len(info["val_scores"]) == 9
    assert info["chosen"] in info["grid"] and info["val_scores"][info["grid"].index(info["chosen"])] == max(info["val_scores"])
    assert P.shape == (g.num_nodes, g.num_classes)
    assert sg.PYGSD_VERSION in info["library"]
