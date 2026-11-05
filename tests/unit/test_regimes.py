"""PR-4 tests: v/p/h/T regimes (Master Prompt §2) and the T11 leakage guard (§1.8, §2.7)."""

import itertools
import math

import numpy as np
import pytest

from edsp.data import GraphData, LabelAccessError, Split
from edsp.methods import run_edsp
from edsp.regimes import DEFAULT_CONFIG, build_inputs
from edsp.regimes import polarity as H
from edsp.regimes import probability as P
from edsp.regimes.priors import track_a_priors, track_b_priors
from edsp.utils.numerics import fit_logreg, randomized_svd, roc_auc


def toy_graph(seed: int = 0, n: int = 90, k: int = 3, signed: bool = False, relations: bool = False) -> GraphData:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    feats = np.hstack([np.eye(k)[y] * 2.0, np.zeros((n, 3))]) + rng.normal(0, 1.0, (n, k + 3))
    rows = []
    for _ in range(4 * n):
        i = int(rng.integers(n))
        same = rng.random() < 0.6
        pool = np.flatnonzero((y == y[i]) if same else (y != y[i]))
        j = int(rng.choice(pool))
        if i != j:
            rows.append((min(i, j), max(i, j)))
    edges = np.unique(np.array(rows), axis=0)
    perm = rng.permutation(n)
    split = Split("0", np.sort(perm[: n // 2]), np.sort(perm[n // 2 : 3 * n // 4]), np.sort(perm[3 * n // 4 :]), "toy")
    sign = None
    if signed:
        sign = np.where(y[edges[:, 0]] == y[edges[:, 1]], 1.0, -1.0) * rng.integers(1, 4, len(edges))
        flip = rng.random(len(edges)) < 0.1
        sign[flip] *= -1
    rel = None
    if relations:
        rel = (y[edges[:, 0]] != y[edges[:, 1]]).astype(int)
        rel[rng.random(len(edges)) < 0.2] ^= 1
    attr = rng.uniform(0, 1, (len(edges), 3))
    return GraphData("toy", n, edges, k, [split], y, features=feats, edge_attr=attr, edge_sign=sign, edge_relation=rel)


def with_test_labels_changed(g: GraphData, seed: int = 1) -> GraphData:
    rng = np.random.default_rng(seed)
    y = g._labels.copy()
    t = g.splits[0].test_idx
    y[t] = (y[t] + rng.integers(1, g.num_classes, len(t))) % g.num_classes
    return GraphData(g.name, g.num_nodes, g.edges, g.num_classes, g.splits, y, features=g.features,
                     edge_attr=g.edge_attr, edge_sign=g.edge_sign, edge_relation=g.edge_relation)


H_SPECS = [{"name": "trainonly", "fill": f} for f in ("b1", "b2", "b3", "b4")] + [
    {"name": "all0"}, {"name": "all1"}, {"name": "learned"}, {"name": "native"}, {"name": "relation"}]
P_SPECS = [{"name": "const", "p0": 0.5}, {"name": "learned"}, {"name": "native", "mode": "noisy-or"},
           {"name": "count"}, {"name": "signmag", "mode": "exp"}]


def _configs():
    for track, h, p in itertools.product(("A", "B"), H_SPECS, P_SPECS):
        if p["name"] == "learned" and h["name"] != "learned":
            continue
        yield {"track": track, "h": h, "p": p}


# ----------------------------------------------------------------------------- T11
@pytest.mark.parametrize("config", list(_configs()), ids=lambda c: f"{c['track']}-{c['h']['name']}{c['h'].get('fill', '')}-{c['p']['name']}")
def test_t11_no_test_label_access_and_invariance(config):
    """T11: every method/regime runs in strict mode, never touches test labels, and
    its output is bit-identical when the test labels are changed."""
    g = toy_graph(signed=True, relations=True)
    run = run_edsp(g, 0, config, max_iter=200)
    assert run.vault.test_accesses_outside_evaluate() == []
    assert not any(r.kind in ("test", "oracle", "descriptive") for r in run.vault.access_log)
    run2 = run_edsp(with_test_labels_changed(g), 0, config, max_iter=200)
    assert np.array_equal(run.probabilities, run2.probabilities)


def test_t11_oracle_is_blocked_without_flag(monkeypatch):
    from edsp.regimes.oracle import h_oracle

    monkeypatch.delenv("ORACLE", raising=False)
    g = toy_graph()
    v = g.vault(0)
    with pytest.raises(LabelAccessError):
        h_oracle(g, v)
    with pytest.raises(PermissionError):
        build_inputs(g, g.vault(0), {"h": {"name": "oracle"}})


def test_oracle_with_flag_is_logged(monkeypatch):
    from edsp.regimes.oracle import h_oracle

    monkeypatch.setenv("ORACLE", "1")
    g = toy_graph()
    v = g.vault(0)
    pol = h_oracle(g, v)
    y = g._labels
    assert np.array_equal(pol.h, (y[g.edges[:, 0]] != y[g.edges[:, 1]]).astype(np.int8))
    assert v.oracle_accessed()


# ------------------------------------------------------------------ polarity
def test_trainonly_fills_and_coverage():
    g = toy_graph()
    v = g.vault(0)
    tr = np.zeros(g.num_nodes, bool)
    tr[g.splits[0].train_idx] = True
    known = tr[g.edges[:, 0]] & tr[g.edges[:, 1]]
    y = g._labels
    het = y[g.edges[:, 0]] != y[g.edges[:, 1]]
    b1 = H.h_trainonly(g, v, "b1")
    assert np.array_equal(b1.h[known], het[known].astype(np.int8)) and not b1.h[~known].any()
    assert H.h_trainonly(g, v, "b2").h[~known].all()
    b3 = H.h_trainonly(g, v, "b3")
    assert len(set(b3.h[~known].tolist())) == 1
    b4 = H.h_trainonly(g, v, "b4")
    assert np.array_equal(b4.keep, known)
    assert math.isclose(b1.diagnostics["coverage"], known.mean())


def test_native_polarity_from_summed_sign():
    g = toy_graph(signed=True)
    assert np.array_equal(H.h_native(g).h, (g.edge_sign < 0).astype(np.int8))


def test_relation_polarity_uses_sum_pi_squared_baseline():
    g = toy_graph(relations=True)
    pol = H.h_relation(g, g.vault(0))
    rel = pol.diagnostics["relations"]
    assert rel[0]["h"] == 0 and rel[1]["h"] == 1  # relation 1 is mostly heterophilic by construction


def test_learned_polarity_selects_on_validation_and_returns_confidence():
    g = toy_graph()
    for track in ("A", "B"):
        pol = H.h_learned(g, g.vault(0), track)
        assert pol.confidence.min() >= 0.5 and pol.confidence.max() <= 1.0
        assert len(pol.diagnostics["val_edge_auc"]) == len(pol.diagnostics["l2_grid"])
        assert np.array_equal(pol.h, (pol.score >= 0.5).astype(np.int8))


# ----------------------------------------------------------------- probability
def test_p_count_weights_are_proportional_to_counts():
    g = toy_graph()
    c = g.edge_attr[:, 0] * 10
    g.edge_attr[:, 0] = c
    p = P.p_count(g, mu=2.0)
    assert np.allclose(-np.log1p(-p), c / 2.0)


def test_p_native_modes_and_signmag():
    g = toy_graph(signed=True)
    S = g.edge_attr
    assert np.allclose(P.p_native(g, "noisy-or"), 1 - np.prod(1 - S, axis=1))
    assert np.allclose(P.p_native(g, "channel:1"), S[:, 1])
    pc = P.p_native(g, "string-combined")
    assert np.all((pc >= P.STRING_PRIOR - 1e-12) & (pc <= 1))
    assert np.allclose(P.p_signmag(g, "linear"), np.abs(g.edge_sign) / np.abs(g.edge_sign).max())
    with pytest.raises(ValueError):
        P.p_const(g, 1.5)


# ---------------------------------------------------------------------- priors
def test_track_a_priors():
    g = toy_graph()
    v, T, _ = track_a_priors(g, g.vault(0), T_L=2.0)
    tr = g.splits[0].train_idx
    assert np.allclose(v.sum(1), 1) and np.all(T[tr] == 2.0)
    others = np.setdiff1d(np.arange(g.num_nodes), tr)
    assert np.allclose(v[others], 1 / g.num_classes) and np.all(T[others] == 0)


def test_track_b_priors_are_distributions_with_bounded_trust():
    g = toy_graph()
    v, T, info = track_b_priors(g, g.vault(0), tau=2.0)
    assert np.allclose(v.sum(1), 1) and np.all(v >= 0)
    assert np.all((T >= 0) & (T <= 2.0 + 1e-12))
    assert info["temperature"] > 0
    v2, T2, _ = track_b_priors(g, g.vault(0), train_onehot=True)
    tr = g.splits[0].train_idx
    assert np.all(np.isinf(T2[tr])) and np.allclose(v2[tr].max(1), 1)


def test_default_config_is_zero_tuning_default():
    assert DEFAULT_CONFIG == {"track": "A", "p": {"name": "const", "p0": 0.5}, "h": {"name": "trainonly", "fill": "b1"}, "T_L": math.inf}


# ------------------------------------------------------------------- numerics
def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.3
    s = np.round(rng.random(200), 1)  # ties
    pos, neg = s[y], s[~y]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert math.isclose(roc_auc(y, s), brute)


def test_logreg_separates_and_svd_reconstructs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = fit_logreg(X, y, 2, l2=0.01)
    assert np.mean(m.predict_proba(X).argmax(1) == y) > 0.95
    A = rng.normal(size=(50, 4)) @ rng.normal(size=(4, 30))
    US = randomized_svd(A, 4)
    assert np.linalg.norm(US @ np.linalg.pinv(US) @ A - A) < 1e-8 * np.linalg.norm(A)


# ------------------------------------------------------------- PR-4b decisions
def heterophilic_noise_graph(seed: int = 0, n: int = 300, k: int = 3, het: float = 0.9) -> GraphData:
    """Edges are heterophilic with probability `het`; features carry no signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    rows = []
    for _ in range(6 * n):
        i = int(rng.integers(n))
        pool = np.flatnonzero((y != y[i]) if rng.random() < het else (y == y[i]))
        j = int(rng.choice(pool))
        if i != j:
            rows.append((min(i, j), max(i, j)))
    edges = np.unique(np.array(rows), axis=0)
    perm = rng.permutation(n)
    split = Split("0", np.sort(perm[: n // 2]), np.sort(perm[n // 2 : 3 * n // 4]), np.sort(perm[3 * n // 4 :]), "toy")
    return GraphData("het-noise", n, edges, k, [split], y, features=rng.normal(size=(n, 4)))


def test_prior_correction_restores_heterophilic_base_rate():
    g = heterophilic_noise_graph()
    bal = H.h_learned(g, g.vault(0), "B", decision="balanced")
    cor = H.h_learned(g, g.vault(0), "B", decision="prior-corrected")
    pi = cor.diagnostics["train_edge_heterophily"]
    assert pi > 0.8
    # uninformative features: the balanced rule labels roughly half the edges heterophilic,
    # the prior-corrected rule recovers the base rate
    assert bal.diagnostics["predicted_heterophilic_fraction"] < 0.75
    assert cor.diagnostics["predicted_heterophilic_fraction"] > 0.95
    # the correction is monotone, so the ranking (and AUC) is unchanged
    order_raw = np.argsort(cor.raw_score, kind="stable")
    assert np.all(np.diff(cor.score[order_raw]) >= -1e-12)


def test_val_threshold_is_selected_from_grid():
    g = heterophilic_noise_graph(1)
    pol = H.h_learned(g, g.vault(0), "A", decision="val-threshold")
    assert pol.diagnostics["threshold"] in np.round(np.linspace(0.05, 0.95, 19), 2)
    assert np.array_equal(pol.h, (pol.score >= pol.diagnostics["threshold"]).astype(np.int8))


def test_learned_fallbacks_and_invalid_decision():
    g = toy_graph()
    y = g._labels.copy()
    tr = g.splits[0].train_idx
    y[tr] = 0  # all train-train edges homophilic
    g2 = GraphData("one-class", g.num_nodes, g.edges, g.num_classes, g.splits, y, features=g.features)
    pol = H.h_learned(g2, g2.vault(0), "A")
    assert "H-ALL0" in pol.diagnostics["status"] and not pol.h.any()
    with pytest.raises(ValueError):
        H.h_learned(g, g.vault(0), "A", decision="nope")
