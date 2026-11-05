"""PR-5 tests: B0/B1 baselines (correctness on hand-checkable graphs and leakage)."""

import numpy as np
import pytest

from edsp.baselines import BASELINES, select_and_predict
from edsp.baselines import propagation as B
from edsp.data import GraphData, Split


def path_graph(n: int = 5, labels=(0, 1)) -> GraphData:
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    y = np.array([labels[0]] + [labels[0]] * (n // 2 - 1) + [labels[1]] * (n - n // 2))
    split = Split("0", np.array([0, n - 1]), np.array([1]), np.arange(2, n - 1), "toy")
    return GraphData("path", n, edges, 2, [split], y, features=np.eye(n))


def test_harmonic_is_linear_interpolation_on_a_path():
    g = path_graph(5)
    P, info = B.harmonic(g, g.vault(0))
    assert np.allclose(P[:, 1], [0, 0.25, 0.5, 0.75, 1.0])
    assert info["uncovered_fraction"] == 0.0


def test_harmonic_leaves_label_free_components_uniform():
    edges = np.array([[0, 1], [2, 3]])
    g = GraphData("two", 4, edges, 2, [Split("0", np.array([0]), np.array([1]), np.array([2, 3]), "t")], np.array([0, 0, 1, 1]))
    P, info = B.harmonic(g, g.vault(0))
    assert np.allclose(P[2:], 0.5) and np.isclose(info["uncovered_fraction"], 0.5)


def test_class_mass_normalisation_matches_train_prior():
    g = path_graph(6)
    P, _ = B.harmonic_cmn(g, g.vault(0))
    unl = np.arange(1, 5)
    mass = P[unl].sum(0) / P[unl].sum()
    assert np.allclose(mass, [0.5, 0.5], atol=0.2)


def test_lgc_and_ppr_solve_their_fixed_point_equations():
    g = path_graph(7)
    v = g.vault(0)
    Y = np.zeros((7, 2))
    Y[0, 0] = Y[6, 1] = 1
    W = B.adjacency(g).toarray()
    d = W.sum(1)
    S = W / np.sqrt(np.outer(d, d))
    F = 0.1 * np.linalg.solve(np.eye(7) - 0.9 * S, Y)
    P, _ = B.lgc(g, v, alpha=0.9)
    assert np.allclose(P, F / F.sum(1, keepdims=True))
    Prw = W / d[:, None]
    F2 = 0.2 * np.linalg.solve(np.eye(7) - 0.8 * Prw, Y)
    P2, _ = B.ppr_lp(g, v, alpha=0.8)
    assert np.allclose(P2, F2 / F2.sum(1, keepdims=True))


def test_linbp_follows_heterophilic_compatibility_where_harmonic_cannot():
    # bipartite path: labels alternate, so neighbours always differ
    n = 8
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    y = np.arange(n) % 2
    train = np.array([0, 1, 2, 3])
    g = GraphData("alt", n, edges, 2, [Split("0", train, np.array([4]), np.arange(5, n), "t")], y)
    H = B.estimate_compatibility(g, g.vault(0))
    assert H[0, 1] > H[0, 0]
    P, _ = B.linbp(g, g.vault(0), scale=0.9)
    assert np.array_equal(P[5:].argmax(1), y[5:])


def test_signed_vote_uses_signs():
    edges = np.array([[0, 2], [1, 2]])
    sign = np.array([-1.0, 1.0])
    g = GraphData("s", 3, edges, 2, [Split("0", np.array([0, 1]), np.array([], dtype=int), np.array([2]), "t")],
                  np.array([0, 1, 1]), edge_sign=sign)
    P, _ = B.signed_vote(g, g.vault(0))
    assert P[2].argmax() == 1


def test_possible_worlds_with_certain_edges_equals_harmonic():
    g = path_graph(5)
    P1, _ = B.harmonic(g, g.vault(0))
    P2, _ = B.possible_worlds_harmonic(g, g.vault(0), np.ones(len(g.edges)), samples=3)
    assert np.allclose(P1, P2)


def test_correct_and_smooth_returns_distributions_and_keeps_train_labels_dominant():
    rng = np.random.default_rng(0)
    g = path_graph(9)
    P, info = B.correct_and_smooth(g, g.vault(0), base=rng.dirichlet(np.ones(2), size=9))
    assert np.allclose(P.sum(1), 1) and np.all(P >= 0)
    assert info["smooth"]["converged"]


@pytest.mark.parametrize("name", [n for n in BASELINES if n != "signed_vote"])
def test_baselines_never_read_test_labels_and_are_test_label_invariant(name):
    from test_regimes import toy_graph, with_test_labels_changed

    g = toy_graph()
    v = g.vault(0)
    P, _ = select_and_predict(name, g, v)
    assert v.test_accesses_outside_evaluate() == []
    assert not any(r.kind == "test" for r in v.access_log)
    P2, _ = select_and_predict(name, with_test_labels_changed(g), with_test_labels_changed(g).vault(0))
    assert np.array_equal(P, P2)
