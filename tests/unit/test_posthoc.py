"""PR-39: Track-B post-hoc comparators (C&S 2-hop, CLP, LinBP-B) on the base predictor of EDSP-B / C&S."""

import itertools

import numpy as np
import pytest
from test_regimes import toy_graph, with_test_labels_changed

from edsp.baselines import propagation as B
from edsp.baselines import select_and_predict
from edsp.baselines.posthoc import (
    CLP_INITS,
    POSTHOC,
    POSTHOC_GRIDS,
    POSTHOC_METHOD,
    clp,
    clp_compatibility,
    linbp_b,
    select_and_predict_posthoc,
    sinkhorn,
)
from edsp.data import GraphData, Split
from edsp.eval.evaluate import evaluate


def planted(kind: str, k: int = 2, n: int = 300, deg: int = 6, noise: float = 0.7, seed: int = 0) -> GraphData:
    """Homophilous (every edge inside a class) or heterophilous (every edge across classes; bipartite
    for k = 2) graph with weakly informative features one_hot(y) + N(0, noise^2); 20/20/60 split."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    rows: set = set()
    while len(rows) < n * deg // 2:
        i = int(rng.integers(n))
        j = int(rng.choice(np.flatnonzero((y == y[i]) if kind == "homophilous" else (y != y[i]))))
        if i != j:
            rows.add((min(i, j), max(i, j)))
    feats = np.eye(k)[y] + rng.normal(0, noise, (n, k))
    perm = rng.permutation(n)
    split = Split("0", np.sort(perm[: n // 5]), np.sort(perm[n // 5 : 2 * n // 5]), np.sort(perm[2 * n // 5 :]), "toy")
    return GraphData(kind, n, np.array(sorted(rows)), k, [split], y, features=feats)


def grid_points(name: str) -> list[dict]:
    g = POSTHOC_GRIDS[name]
    return [dict(zip(g, v, strict=True)) for v in itertools.product(*g.values())]


def test_grids_have_the_budget_of_cs_and_edsp_tuned():
    cs = len(list(itertools.product(*B.GRIDS["correct_and_smooth"].values())))
    assert cs == 9
    assert {n: len(grid_points(n)) for n in POSTHOC} == {"cs_2hop": 9, "clp": 9, "linbp_b": 9}
    assert POSTHOC_GRIDS["cs_2hop"] == B.GRIDS["correct_and_smooth"] and POSTHOC_GRIDS["linbp_b"] == B.GRIDS["linbp"]
    assert set(POSTHOC_METHOD.values()) == set(POSTHOC)


@pytest.mark.parametrize("name", list(POSTHOC))
@pytest.mark.parametrize("kind", ["homophilous", "heterophilous"])
def test_outputs_are_distributions_over_the_whole_grid(name, kind):
    g = planted(kind, k=3, n=120)
    v = g.vault(0)
    P, info = select_and_predict_posthoc(name, g, v)
    assert P.shape == (g.num_nodes, g.num_classes) and np.all(np.isfinite(P)) and np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert info["chosen"] in info["grid"] and len(info["grid"]) == len(info["val_scores"]) == 9
    Z = B.feature_lr(g, v)[0]
    for c in grid_points(name):
        Q = POSTHOC[name](g, v, **c, base=Z)[0]
        assert np.all(Q >= 0) and np.allclose(Q.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", list(POSTHOC))
def test_label_isolation(name):
    g = toy_graph()
    v = g.vault(0)
    P, _ = select_and_predict_posthoc(name, g, v)
    assert v.test_accesses_outside_evaluate() == [] and not any(r.kind == "test" for r in v.access_log)
    val = [r for r in v.access_log if r.kind == "val"]
    assert val and all(r.allowed and r.caller.startswith("edsp.selection") for r in val)
    # test labels never matter
    g2 = with_test_labels_changed(g)
    assert np.array_equal(P, select_and_predict_posthoc(name, g2, g2.vault(0))[0])
    # validation labels only select among grid points: every grid point is invariant to them
    y = g._labels.copy()
    va = g.splits[0].val_idx
    y[va] = (y[va] + 1) % g.num_classes
    g3 = GraphData(g.name, g.num_nodes, g.edges, g.num_classes, g.splits, y, features=g.features)
    Z = B.feature_lr(g, v)[0]
    for c in grid_points(name):
        assert np.array_equal(POSTHOC[name](g, v, **c, base=Z)[0], POSTHOC[name](g3, g3.vault(0), **c, base=Z)[0])


@pytest.mark.parametrize("name", list(POSTHOC))
def test_deterministic(name):
    g = planted("heterophilous", k=3, n=120)
    a, ia = select_and_predict_posthoc(name, g, g.vault(0))
    b, ib = select_and_predict_posthoc(name, g, g.vault(0))
    assert np.array_equal(a, b) and ia["chosen"] == ib["chosen"] and ia["val_scores"] == ib["val_scores"]


# ------------------------------------------------------------------ C&S 2-hop
def _cs_reference(g, v, ac, as_, Z, S):
    """The PR-5 C&S (unchanged code before PR-39) with an arbitrary propagation matrix S."""
    Y, idx, _ = B._one_hot_train(g, v)
    E0 = np.zeros_like(Z)
    E0[idx] = Y[idx] - Z[idx]
    E, _ = B._iterate(lambda E: ac * (S @ E) + (1 - ac) * E0, E0.copy())
    sigma = float(np.abs(E0[idx]).sum(axis=1).mean())
    norm = np.abs(E).sum(axis=1, keepdims=True)
    scale = np.divide(sigma, norm, out=np.zeros_like(norm), where=norm > 0)
    G0 = np.clip(Z + scale * E, 0.0, None)
    G0[idx] = Y[idx]
    G, _ = B._iterate(lambda G: as_ * (S @ G) + (1 - as_) * G0, G0.copy())
    return B._to_probs(G)


def test_one_hop_cs_is_bit_identical_and_two_hop_applies_s_twice():
    g = planted("heterophilous", k=3, n=150)
    v = g.vault(0)
    Z = B.feature_lr(g, v)[0]
    S = B._sym_norm(B.adjacency(g))
    for ac, as_ in ((0.5, 0.8), (0.95, 0.5)):
        assert np.array_equal(B.correct_and_smooth(g, v, ac, as_, base=Z)[0], _cs_reference(g, v, ac, as_, Z, S))
        P2, info = POSTHOC["cs_2hop"](g, v, ac, as_, base=Z)
        assert info["hops"] == 2
        assert np.allclose(P2, _cs_reference(g, v, ac, as_, Z, (S @ S).toarray()), atol=1e-8)


def test_cs_2hop_never_forms_the_two_hop_matrix(monkeypatch):
    """S is only ever applied to dense (n, k) blocks, as in Lim et al.'s code: no A^2 / S^2 is built,
    so memory equals 1-hop C&S and no nnz(A^2) guard is needed."""
    class VectorOnly:
        def __init__(self, S):
            self.S, self.calls = S, 0

        def __matmul__(self, X):
            assert isinstance(X, np.ndarray) and X.ndim == 2, type(X)
            self.calls += 1
            return self.S @ X

    seen = []
    orig = B._sym_norm
    monkeypatch.setattr(B, "_sym_norm", lambda W: seen.append(VectorOnly(orig(W))) or seen[-1])
    g = planted("homophilous", n=100)
    Z = B.feature_lr(g, g.vault(0))[0]
    a = POSTHOC["cs_2hop"](g, g.vault(0), base=Z)[0]
    assert seen and seen[0].calls > 0
    monkeypatch.setattr(B, "_sym_norm", orig)
    assert np.array_equal(a, POSTHOC["cs_2hop"](g, g.vault(0), base=Z)[0])


# ------------------------------------------------------------------ CLP
def test_sinkhorn_is_doubly_stochastic():
    H, info = sinkhorn(np.random.default_rng(0).uniform(0.01, 5, (4, 4)))
    assert info["converged"] and np.allclose(H.sum(0), 1) and np.allclose(H.sum(1), 1)


def test_clp_compatibility_sees_heterophily_and_uses_train_labels_only():
    g = planted("heterophilous", k=2)
    Z = B.feature_lr(g, g.vault(0))[0]
    v = g.vault(0)  # fresh vault: only clp_compatibility's accesses are logged
    H, B0, info = clp_compatibility(g, v, Z)
    assert np.allclose(H.sum(0), 1) and np.allclose(H.sum(1), 1) and info["sinkhorn"]["converged"]
    assert H[0, 1] > H[0, 0] and H[1, 0] > H[1, 1]
    idx, y = v.train_labels()
    assert np.array_equal(B0[idx], np.eye(2)[y])
    assert not any(r.kind in ("val", "test") for r in v.access_log)


def test_clp_layer_matches_the_official_edge_formula():
    """One layer against an edge loop over CLP's get_edge_weight / LabelPropagationHeterophily:
    w_ij[c] = B0[i, c] (B0[j] H)[c]; X <- clamp(alpha L1norm(sum_j w_ij X_j) + (1 - alpha) X0)."""
    g = planted("heterophilous", k=3, n=40, deg=4)
    v = g.vault(0)
    Z = B.feature_lr(g, v)[0]
    H, B0, _ = clp_compatibility(g, v, Z)
    Y = B._one_hot_train(g, v)[0]
    for init, X0 in (("base", Z), ("base+train", B0), ("train", Y)):
        agg = np.zeros_like(Z)
        for a, b in g.edges:
            for i, j in ((a, b), (b, a)):  # i receives from j
                agg[i] += B0[i] * (B0[j] @ H) * X0[j]
        agg /= np.maximum(agg.sum(1, keepdims=True), 1e-12)
        ref = np.clip(0.5 * agg + 0.5 * X0, 0, 1)
        assert np.allclose(clp(g, v, alpha=0.5, init=init, base=Z, layers=1)[0], B._to_probs(ref), atol=1e-12)
    with pytest.raises(ValueError):
        clp(g, v, init="echo", base=Z)
    assert set(CLP_INITS) == {"base", "base+train", "train"}


def test_clp_messages_are_class_symmetric_under_exact_bipartite_compatibility(monkeypatch):
    """Ĥ enters CLP only through the edge weights and the class-c message carries the neighbour's
    class-c belief: with k = 2 and an exactly anti-diagonal Ĥ, start matrix B0 is a fixed point."""
    import edsp.baselines.posthoc as PH

    g = planted("heterophilous", k=2, n=120)
    Z = B.feature_lr(g, g.vault(0))[0]
    real = PH.clp_compatibility

    def exact(graph, vault, base):
        _, B0, info = real(graph, vault, base)
        return np.array([[0.0, 1.0], [1.0, 0.0]]), B0, info

    monkeypatch.setattr(PH, "clp_compatibility", exact)
    _, B0, _ = real(g, g.vault(0), Z)
    for a in (0.1, 0.5, 0.9):
        assert np.allclose(PH.clp(g, g.vault(0), alpha=a, init="base+train", base=Z)[0], B0, atol=1e-12)


# ------------------------------------------------------------------ LinBP-B
def test_linbp_b_with_uninformative_base_is_linbp_and_otherwise_uses_it():
    g = planted("heterophilous", k=3, n=150)
    v = g.vault(0)
    U = np.full((g.num_nodes, 3), 1 / 3)
    for s in (0.1, 0.9):
        assert np.array_equal(linbp_b(g, v, scale=s, base=U)[0], B.linbp(g, v, scale=s)[0])
    Z = B.feature_lr(g, v)[0]
    P, info = linbp_b(g, v, scale=0.05, base=Z)
    unl = np.setdiff1d(np.arange(g.num_nodes), v.train_idx)
    # tiny eps: beliefs stay close to the explicit (base) beliefs on non-train nodes
    assert info["seeded_with_base"] and np.mean(P[unl].argmax(1) == Z[unl].argmax(1)) > 0.9


# ------------------------------------------------------------------ behaviour on toy graphs
def _accuracies(g):
    v = g.vault(0)
    Z = B.feature_lr(g, v)[0]
    acc = {"base": evaluate(v, Z)["accuracy"],
           "correct_and_smooth": evaluate(v, select_and_predict("correct_and_smooth", g, v)[0])["accuracy"]}
    for m in POSTHOC:
        acc[m] = evaluate(v, select_and_predict_posthoc(m, g, v, base=Z)[0])["accuracy"]
    return acc


@pytest.mark.parametrize("k", [2, 3])
def test_homophilous_toy_all_improve_on_the_base_predictor(k):
    a = _accuracies(planted("homophilous", k=k))
    for m in POSTHOC:
        assert a[m] >= a["base"] + 0.1 and a[m] >= 0.95, (m, a)


@pytest.mark.parametrize("k", [2, 3])
def test_heterophilous_toy_cs_collapses_linbp_b_gains_clp_holds(k):
    a = _accuracies(planted("heterophilous", k=k))
    assert a["correct_and_smooth"] < a["base"] - 0.1, a     # 1-hop C&S smooths across classes
    assert a["linbp_b"] > a["base"] + 0.1, a                 # compatibility-aware propagation gains
    assert a["clp"] >= a["base"] - 0.03, a                   # CLP does not collapse (stays near its base)
    if k == 2:                                               # bipartite: two hops connect same-class nodes
        assert a["cs_2hop"] > a["base"] + 0.1, a
