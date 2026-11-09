"""Train/val label noise (§5.5, H6): test labels stay clean, isolation holds."""

import dataclasses
import math

import numpy as np
import pytest

from edsp.data import label_noise
from edsp.data.graph import GraphData, Split
from edsp.data.vault import LabelAccessError, LabelVault
from edsp.eval.evaluate import evaluate
from edsp.methods import run_edsp


def _graph(k: int = 3, n: int = 3000) -> GraphData:
    rng = np.random.default_rng(0)
    y = rng.integers(0, k, n)
    idx = rng.permutation(n)
    splits = [Split(str(s), np.sort(np.roll(idx, 300 * s)[:1200]), np.sort(np.roll(idx, 300 * s)[1200:1800]),
                    np.sort(np.roll(idx, 300 * s)[1800:]), "t") for s in range(2)]
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return GraphData("toy", n, edges, k, splits, y)


def test_parse_and_tag():
    assert label_noise.parse("uniform:0.3") == ("uniform", 0.3)
    assert label_noise.tag("pairwise:0.10") == "ln-pairwise-0.1"
    for bad in ("uniform", "gauss:0.1", "uniform:1.5"):
        with pytest.raises(ValueError):
            label_noise.parse(bad)


def test_clean_graph_is_unchanged():
    g = _graph()
    v = g.vault(0)
    assert v.noise_info is None and v._observed is None


@pytest.mark.parametrize("spec", ["uniform:0.3", "pairwise:0.3"])
def test_noise_hits_train_and_val_only(spec):
    g = _graph()
    clean = g.vault(0)
    noisy = dataclasses.replace(g, label_noise=(spec, 7)).vault(0)
    _, ytr = clean.train_labels()
    _, ntr = noisy.train_labels()
    frac = float(np.mean(ytr != ntr))
    assert 0.25 < frac < 0.35
    assert noisy.noise_info["flipped_train"] == int(np.sum(ytr != ntr))
    # test labels are the true labels: a perfect predictor of the true labels scores 1 on the noisy vault
    p = np.eye(g.num_classes)[g._labels]
    assert math.isclose(evaluate(noisy, p)["accuracy"], 1.0)
    if spec.startswith("pairwise"):
        _, yorig = clean.train_labels()
        changed = ntr != yorig
        assert np.all(ntr[changed] == (yorig[changed] + 1) % g.num_classes)


def test_noise_is_deterministic_and_split_dependent():
    g = dataclasses.replace(_graph(), label_noise=("uniform:0.4", 1))
    a = g.vault(0).train_labels()[1]
    assert np.array_equal(a, g.vault(0).train_labels()[1])
    b = dataclasses.replace(g, label_noise=("uniform:0.4", 2)).vault(0).train_labels()[1]
    assert not np.array_equal(a, b)
    assert g.vault(0).noise_info["flipped_val"] > 0
    assert g.content_hash() != _graph().content_hash()


def test_k2_uniform_equals_pairwise():
    g = _graph(k=2)
    u = dataclasses.replace(g, label_noise=("uniform:0.3", 3)).vault(1).train_labels()[1]
    w = dataclasses.replace(g, label_noise=("pairwise:0.3", 3)).vault(1).train_labels()[1]
    assert np.array_equal(u, w)


def test_observed_must_match_test_labels_and_methods_cannot_read_descriptive():
    y = np.array([0, 1, 0, 1])
    obs = y.copy()
    obs[3] = 0  # node 3 is a test node
    with pytest.raises(ValueError):
        LabelVault(y, np.array([0]), np.array([1]), np.array([2, 3]), num_classes=2, _observed=obs)
    v = dataclasses.replace(_graph(), label_noise=("uniform:0.2", 0)).vault(0)
    with pytest.raises(LabelAccessError):
        v.descriptive_labels()  # called from a test module, not an allowed descriptive module


def test_finite_trust_runs_under_noise():
    g = dataclasses.replace(_graph(), label_noise=("uniform:0.4", 0))
    v = g.vault(0)
    for T in (math.inf, 1.0):
        r = run_edsp(g, config={"track": "A", "p": {"name": "const", "p0": 0.5},
                                "h": {"name": "trainonly", "fill": "b1"}, "T_L": T}, vault=v)
        assert np.allclose(r.probabilities.sum(axis=1), 1.0)
