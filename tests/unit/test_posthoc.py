"""PR-39: Track-B post-hoc comparators (C&S 2-hop, CLP, LinBP-B and the named variants C&S-ref,
C&S-2hop-ref, CLP-ES) on the base predictor of EDSP-B / C&S."""

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
    clp_es,
    cs_reference_raw,
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
    assert {n: len(grid_points(n)) for n in POSTHOC} == dict.fromkeys(
        ("cs_2hop", "clp", "linbp_b", "cs_ref", "cs_2hop_ref", "clp_es"), 9)
    for n in ("cs_2hop", "cs_ref", "cs_2hop_ref"):
        assert POSTHOC_GRIDS[n] == B.GRIDS["correct_and_smooth"]
    assert POSTHOC_GRIDS["linbp_b"] == B.GRIDS["linbp"] and POSTHOC_GRIDS["clp_es"] == POSTHOC_GRIDS["clp"]
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
    # validation labels only select among grid points (and, for clp_es, layers): with the selection
    # fixed, every output is invariant to them
    y = g._labels.copy()
    va = g.splits[0].val_idx
    y[va] = (y[va] + 1) % g.num_classes
    g3 = GraphData(g.name, g.num_nodes, g.edges, g.num_classes, g.splits, y, features=g.features)
    Z = B.feature_lr(g, v)[0]
    for c in grid_points(name):
        if name == "clp_es":
            layer = POSTHOC[name](g, v, **c, base=Z)[1]["selected_layer"]
            a = clp(g, v, **c, base=Z, layers=layer)[0]
            assert np.array_equal(a, clp(g3, g3.vault(0), **c, base=Z, layers=layer)[0])
            assert np.array_equal(a, POSTHOC[name](g, v, **c, base=Z)[0])
        else:
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


# ------------------------------------------------------------------ reference C&S (PR-39b)
def _random_graph_with_tail(seed: int = 3, n0: int = 60, k: int = 3):
    """Random graph + a 25-node path (tiny propagated errors: the autoscale cap at 1000 triggers)
    + a 3-node label-free component (zero error: infinite scale)."""
    rng = np.random.default_rng(seed)
    e = rng.integers(0, n0, (200, 2))
    e = np.unique(np.sort(e[e[:, 0] != e[:, 1]], axis=1), axis=0)
    e = np.concatenate([e, [[i, i + 1] for i in range(n0 - 1, n0 + 24)], [[n0 + 26, n0 + 27], [n0 + 27, n0 + 28]]])
    n = n0 + 29
    tr = np.sort(rng.choice(n0, 20, replace=False))
    rest = np.setdiff1d(np.arange(n), tr)
    g = GraphData("r", n, e, k, [Split("0", tr, rest[:20], rest[20:], "t")], rng.integers(0, k, n))
    return g, rng.dirichlet(np.ones(k), n)


def test_cs_ref_reproduces_pyg_correct_and_smooth():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn.models")
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    g, Z = _random_graph_with_tail()
    v = g.vault(0)
    idx, yt = v.train_labels()
    ei = torch.tensor(np.concatenate([g.edges, g.edges[:, ::-1]]).T.copy())
    # float64 normalised weights (PyG's default gcn_norm weights are float32)
    ei, ew = gcn_norm(ei, torch.ones(ei.size(1), dtype=torch.float64), num_nodes=g.num_nodes, add_self_loops=False)
    S = B._sym_norm(B.adjacency(g))
    Y = B._one_hot_train(g, v)[0]
    for a1, a2 in ((0.5, 0.8), (0.95, 0.5), (0.8, 0.95)):
        cs = pyg.CorrectAndSmooth(50, a1, 50, a2, autoscale=True)
        mask, yy = torch.tensor(idx), torch.tensor(yt)
        ref = cs.smooth(cs.correct(torch.tensor(Z), yy, mask, ei, ew), yy, mask, ei, ew).numpy()
        G, info = cs_reference_raw(lambda X: S @ X, Z, Y, idx, a1, a2)
        assert np.abs(G - ref).max() < 1e-6 and info["autoscale_capped_rows"] > 0
        assert np.allclose(POSTHOC["cs_ref"](g, v, a1, a2, base=Z)[0], B._to_probs(ref), atol=1e-6)


def _lim_double_correlation_autoscale(S, Z, Y, idx, a1, a2, num_hops, layers=50):
    """Line-by-line numpy transcription of Lim et al.'s correct_smooth.double_correlation_autoscale
    (CUAI/Non-Homophily-Large-Scale@82f8f05) with label_idx = residual_idx = train."""
    y = np.zeros_like(Z)
    y[idx] = Y[idx] - Z[idx]
    result = y.copy()
    for _ in range(layers):
        for _ in range(num_hops):
            result = S @ result
        result = np.clip(a1 * result + (1 - a1) * y, -1.0, 1.0)
    orig_diff = np.abs(y[idx]).sum() / len(idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid_scale = orig_diff / np.abs(result).sum(axis=1, keepdims=True)
        resid_scale[np.isinf(resid_scale)] = 1.0
        resid_scale[resid_scale > 1000] = 1.0
        res_result = Z + resid_scale * result
    res_result[np.isnan(res_result)] = Z[np.isnan(res_result)]
    y = res_result.copy()
    y[idx] = Y[idx]
    result = y.copy()
    for _ in range(layers):
        for _ in range(num_hops):
            result = S @ result
        result = np.clip(a2 * result + (1 - a2) * y, 0.0, 1.0)
    return result


@pytest.mark.parametrize("hops", [1, 2])
def test_reference_cs_follows_lims_code(hops):
    g, Z = _random_graph_with_tail(seed=5)
    v = g.vault(0)
    S = B._sym_norm(B.adjacency(g))
    Y, idx, _ = B._one_hot_train(g, v)
    for a1, a2 in ((0.5, 0.95), (0.95, 0.8)):
        ref = _lim_double_correlation_autoscale(S.toarray(), Z, Y, idx, a1, a2, hops)
        name = "cs_ref" if hops == 1 else "cs_2hop_ref"
        P, info = POSTHOC[name](g, v, a1, a2, base=Z)
        assert info["hops"] == hops and info["layers"] == 50
        assert np.allclose(P, B._to_probs(ref), atol=1e-10)


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


def test_clp_es_picks_the_best_validation_layer():
    from edsp.selection import validation_score

    g = planted("heterophilous", k=2, n=150)
    Z = B.feature_lr(g, g.vault(0))[0]
    v = g.vault(0)
    P, info = clp_es(g, v, alpha=0.5, init="train", base=Z)
    scores = info["layer_val_scores"]
    assert len(scores) == 50 and 1 <= info["selected_layer"] <= 50
    assert info["selected_layer"] == int(np.argmax(scores)) + 1  # first best layer
    assert np.array_equal(P, clp(g, v, alpha=0.5, init="train", base=Z, layers=info["selected_layer"])[0])
    assert np.isclose(validation_score(v, P), max(scores))
    assert all(r.caller.startswith("edsp.selection") for r in v.access_log if r.kind == "val")
    _, sel = select_and_predict_posthoc("clp_es", g, g.vault(0), base=Z)
    assert len(sel["selected_layers"]) == 9 and sel["selected_layer"] == sel["selected_layers"][sel["grid"].index(sel["chosen"])]


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
    for m in ("clp", "clp_es"):                              # CLP does not collapse (stays near its base)
        assert a[m] >= a["base"] - 0.03, (m, a)
    if k == 2:                                               # bipartite: two hops connect same-class nodes
        assert a["cs_2hop"] > a["base"] + 0.1 and a["cs_2hop_ref"] > a["base"] + 0.1, a
