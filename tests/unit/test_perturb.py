"""PERTURBED-REAL mixing generator (edsp/data/perturb.py)."""

import numpy as np
import pytest
from test_regimes import toy_graph

from edsp.data import stats as S
from edsp.data.perturb import MODES, parse, perturb


def _h(g):
    return S.edge_homophily(g.edges, g._labels)


@pytest.mark.parametrize("h", [0.0, 0.3, 1.0])
def test_global_mixing_hits_target_and_keeps_nodes_labels_splits(h):
    g = toy_graph(n=300, k=3)
    q = perturb(g, f"global:h={h}", seed=1)
    assert abs(_h(q) - h) < 0.08
    assert q.num_nodes == g.num_nodes and q.splits is g.splits
    assert np.array_equal(q._labels, g._labels)
    assert q.meta["perturbed_real"] and q.edge_sign is None
    assert abs(len(q.edges) - len(g.edges)) < 0.1 * len(g.edges)


def test_permutation_mode_concentrates_heterophily_on_one_class():
    g = toy_graph(n=300, k=4)
    q = perturb(g, "permutation:h=0", seed=0)
    y = q._labels
    H = S.compatibility_matrix(q.edges, y, 4)
    assert np.allclose(np.diag(H), 0) and np.all(H.max(axis=1) > 0.6)  # one dominant off-diagonal class per row
    assert S.label_informativeness(q.edges, y, 4) > 0.5


def test_every_mode_runs_is_deterministic_and_heterogeneous_modes_spread_local_homophily():
    g = toy_graph(n=400, k=3)
    stds = {}
    for mode in MODES:
        a, b = perturb(g, mode, seed=3), perturb(g, mode, seed=3)
        assert np.array_equal(a.edges, b.edges)
        y = a._labels
        stds[mode] = np.nanstd(S.local_homophily(a.edges, y, a.num_nodes))
    assert stds["beta"] > stds["global"] and stds["regional"] > stds["global"]


def test_parse_rejects_unknown_modes():
    assert parse("beta:m=0.2,c=5") == ("beta", {"m": 0.2, "c": 5.0})
    with pytest.raises(ValueError):
        parse("swirl")
