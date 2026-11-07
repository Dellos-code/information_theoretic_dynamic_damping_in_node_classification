"""Track-B GNN baselines (edsp/baselines/gnn.py)."""

import importlib.util

import numpy as np
import pytest
from test_regimes import toy_graph, with_test_labels_changed

torch_ok = importlib.util.find_spec("torch") is not None
if torch_ok:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - broken local installs count as unavailable
        torch_ok = False
pytestmark = pytest.mark.skipif(not torch_ok, reason="torch is unavailable in this environment")


@pytest.mark.parametrize("name", ["mlp", "gcn", "sage", "gat", "gprgnn", "bwgnn"])
def test_each_gnn_learns_the_toy_graph_and_ignores_test_labels(name, monkeypatch):
    from edsp.baselines import gnn

    monkeypatch.setattr(gnn, "MAX_EPOCHS", 60)
    g = toy_graph()
    v = g.vault(0)
    grid = {k: vals[0] for k, vals in gnn.GNN_GRIDS[name].items()}
    P, info = gnn.train_gnn(name, g, v, seed=0, **grid)
    assert P.shape == (g.num_nodes, g.num_classes) and np.allclose(P.sum(1), 1, atol=1e-5)
    idx, _ = v.train_labels()
    train_acc = float(np.mean(P[idx].argmax(1) == g._labels[idx]))
    assert train_acc > 0.5, (name, train_acc)
    assert v.test_accesses_outside_evaluate() == []
    g2 = with_test_labels_changed(g)
    P2, _ = gnn.train_gnn(name, g2, g2.vault(0), seed=0, **grid)
    assert np.allclose(P, P2)


def test_gnn_grids_have_nine_points():
    from edsp.baselines.gnn import GNN_GRIDS

    for grid in GNN_GRIDS.values():
        assert np.prod([len(v) for v in grid.values()]) == 9
