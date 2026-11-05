"""PR-2 tests: canonicalisation, statistics, label isolation, loaders on fixtures.

No network access: loader tests write fixture files into a temporary
EDSP_DATA_DIR, which acquisition.fetch treats as a cache hit.
"""

import json
import math
import pickle
import zipfile
from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sp

from edsp.data import (
    GraphData,
    LabelAccessError,
    LabelVault,
    Split,
    canonicalize_edges,
    random_splits,
)
from edsp.data import stats as S
from edsp.data.registry import datasets_with_loader, registry


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("EDSP_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EDSP_ACQUISITION_LOG", str(tmp_path / "acq.jsonl"))
    yield


# ------------------------------------------------------------- canonicalisation
def test_canonicalize_removes_loops_and_duplicates_and_sums_signs():
    edges = np.array([[0, 1], [1, 0], [2, 2], [1, 2], [2, 1], [0, 1]])
    sign = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    out, _, s, rep = canonicalize_edges(edges, 3, edge_sign=sign)
    assert out.tolist() == [[0, 1], [1, 2]]
    assert rep["self_loops_removed"] == 1 and rep["duplicates_merged"] == 3
    assert s.tolist() == [1.0, 0.0]
    assert rep["sign_conflicting_pairs"] == 2


def test_canonicalize_counts_conflicting_attributes():
    edges = np.array([[0, 1], [1, 0], [1, 2]])
    attr = np.array([[0.1, 0.2], [0.1, 0.3], [0.5, 0.5]])
    out, a, _, rep = canonicalize_edges(edges, 3, edge_attr=attr)
    assert out.tolist() == [[0, 1], [1, 2]]
    assert a.tolist() == [[0.1, 0.2], [0.5, 0.5]]
    assert rep["conflicting_attr_merges"] == 1


def test_random_splits_are_disjoint_stratified_and_skip_negative_labels():
    y = np.array([0] * 10 + [1] * 10 + [-1] * 3)
    for s in random_splits(y, 3, (0.6, 0.2, 0.2), 0):
        allidx = np.concatenate([s.train_idx, s.val_idx, s.test_idx])
        assert len(allidx) == len(np.unique(allidx)) == 20
        assert not np.isin(np.arange(20, 23), allidx).any()
        assert np.bincount(y[s.train_idx]).tolist() == [6, 6]


# ------------------------------------------------------------------ statistics
def _path_graph():
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    return edges, y


def test_homophily_measures_on_hand_computed_path():
    edges, y = _path_graph()
    assert math.isclose(S.edge_homophily(edges, y), 2 / 3)
    assert math.isclose(S.node_homophily(edges, y, 4), 0.75)
    assert math.isclose(S.adjusted_homophily(edges, y, 2), 1 / 3)


def test_label_informativeness_equals_mutual_information_ratio():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n, k = 40, int(rng.integers(2, 6))
        edges = rng.integers(0, n, size=(120, 2))
        edges, _, _, _ = canonicalize_edges(edges, n)
        y = rng.integers(0, k, n)
        counts = S.class_pair_counts(edges, y, k)
        p = counts / counts.sum()
        pb = p.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(np.where(p > 0, p * np.log(p / np.outer(pb, pb)), 0.0))
            h = -np.nansum(np.where(pb > 0, pb * np.log(pb), 0.0))
        assert math.isclose(S.label_informativeness(edges, y, k), mi / h, rel_tol=1e-9, abs_tol=1e-12)


def test_li_is_one_for_perfectly_homophilous_and_bipartite_k2():
    edges = np.array([[0, 1], [2, 3]])
    assert math.isclose(S.label_informativeness(edges, np.array([0, 0, 1, 1]), 2), 1.0)
    assert math.isclose(S.label_informativeness(edges, np.array([0, 1, 0, 1]), 2), 1.0)
    assert math.isclose(S.adjusted_homophily(edges, np.array([0, 1, 0, 1]), 2), -1.0)


def test_compatibility_matrix_rows_and_distances():
    edges, y = _path_graph()
    H = S.compatibility_matrix(edges, y, 2)
    assert np.allclose(H.sum(1), 1)
    assert np.allclose(H, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    d = S.compatibility_distances(np.eye(3))
    assert d["to_identity"] == 0 and d["to_uniform_heterophily"] > 0


def _toy_graph(sign=None) -> GraphData:
    edges, y = _path_graph()
    splits = [Split("0", np.array([0, 3]), np.array([1]), np.array([2]), "toy")]
    return GraphData("toy", 4, edges, 2, splits, y, edge_sign=sign)


def test_dataset_stats_end_to_end_and_train_view():
    st = S.dataset_stats(_toy_graph(np.array([1.0, -1.0, 1.0])))
    assert st["structure"]["components"] == 1
    assert st["labels"]["train_label_view"]["train_train_edges"] == 0
    assert st["labels"]["signed"]["sign_label_agreement"] == 1.0
    assert st["dataset_hash"].startswith("sha256:")


# ---------------------------------------------------------------- label vault
def test_vault_blocks_test_and_val_labels_outside_allowed_modules():
    v = _toy_graph().vault(0)
    idx, y = v.train_labels()
    assert idx.tolist() == [0, 3] and y.tolist() == [0, 1]
    with pytest.raises(LabelAccessError):
        v.test_labels()
    with pytest.raises(LabelAccessError):
        v.val_labels()
    with pytest.raises(LabelAccessError):
        v.descriptive_labels()
    assert len(v.test_accesses_outside_evaluate()) == 2


def test_vault_allows_evaluate_and_selection():
    from edsp.eval.evaluate import evaluate
    from edsp.selection import validation_accuracy

    v = _toy_graph().vault(0)
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert evaluate(v, probs)["accuracy"] == 1.0
    assert validation_accuracy(v, probs) == 1.0
    assert v.test_accesses_outside_evaluate() == []


def test_vault_non_strict_logs_instead_of_raising_and_labels_are_read_only():
    v = _toy_graph().vault(0, strict=False)
    v.test_labels()
    assert v.test_accesses_outside_evaluate()[0].caller == __name__
    with pytest.raises(ValueError):
        v._labels[0] = 5


def test_vault_rejects_overlapping_splits():
    with pytest.raises(ValueError):
        LabelVault(np.zeros(3, dtype=int), [0, 1], [1], [2], num_classes=1)


# ----------------------------------------------------------------- registry
def test_registry_matches_loaders():
    from edsp.data.loaders import LOADERS

    assert set(datasets_with_loader()) == set(LOADERS)
    for name in LOADERS:
        entry = registry()["datasets"][name]
        assert entry["routes"] and all("{file}" in r for r in entry["routes"])


# ------------------------------------------------------------ loader fixtures
def _cache(tmp_path):
    return tmp_path / "cache"


def test_planetoid_loader_on_fixture(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "planetoid"
    d.mkdir(parents=True)
    k, f = 3, 4
    x = sp.csr_matrix(np.eye(2, f))
    y = np.eye(k)[[0, 1]]
    allx = sp.csr_matrix(np.ones((4, f)))
    ally = np.eye(k)[[0, 1, 2, 0]]
    tx = sp.csr_matrix(np.full((2, f), 2.0))
    ty = np.eye(k)[[2, 1]]
    test_index = [5, 4]
    graph = defaultdict(list, {0: [1, 1, 0], 1: [2], 2: [3], 3: [4], 4: [5], 5: [0]})
    for suffix, obj in {"x": x, "y": y, "allx": allx, "ally": ally, "tx": tx, "ty": ty, "graph": graph}.items():
        with open(d / f"ind.cora.{suffix}", "wb") as fh:
            pickle.dump(obj, fh)
    (d / "ind.cora.test.index").write_text("\n".join(map(str, test_index)))
    g = load("cora")
    assert g.num_nodes == 6 and g.num_classes == 3
    assert g.meta["canonicalization"]["self_loops_removed"] == 1
    assert len(g.edges) == 6
    pub = g.splits[0]
    assert pub.train_idx.tolist() == [0, 1] and pub.test_idx.tolist() == [4, 5]
    y_all = g.descriptive_vault()._labels
    # torch_geometric convention: x[test_index] = x[sorted_test_index], i.e. the
    # node listed first in test.index (5) receives row 0 of tx/ty.
    assert y_all[5] == 2 and y_all[4] == 1


def test_heterophilous_npz_loader_on_fixture(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "heterophilous"
    d.mkdir(parents=True)
    n = 6
    masks = np.zeros((10, n), dtype=bool)
    tr, va, te = masks.copy(), masks.copy(), masks.copy()
    tr[:, :3], va[:, 3], te[:, 4:] = True, True, True
    np.savez(d / "roman_empire.npz", node_features=np.ones((n, 2), np.float32), node_labels=np.arange(n) % 2,
             edges=np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]), train_masks=tr, val_masks=va, test_masks=te)
    g = load("roman-empire")
    assert len(g.splits) == 10 and g.splits[0].test_idx.tolist() == [4, 5]
    assert len(g.edges) == 6


def test_pajek_parser_and_sampson_loader_on_fixture(tmp_path):
    from edsp.data.loaders import load

    lines = [
        "*Network Sampson", "*Vertices 2", ' 1 "A"', ' 2 "B"', "*Arcs", " 1 2 1",
        "*Network Sampson_T4", "*Vertices 3", ' 1 "A"', ' 2 "B"', ' 3 "C"',
        "*Arcs", " 1 2 1", " 2 1 1", " 2 3 -1", " 3 2 1", "*Edges", "",
        "*Partition Sampson_factions_T4", "*Vertices 3", "1", "1", "2",
    ]
    text = "\n".join(lines)
    d = _cache(tmp_path) / "sampson"
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "Sampson.zip", "w") as z:
        z.writestr("Sampson.paj", text)
    g = load("sampson")
    assert g.num_nodes == 3 and g.num_classes == 2
    assert g.edges.tolist() == [[0, 1], [1, 2]]
    assert g.edge_sign.tolist() == [2.0, 0.0]
    assert g.meta["canonicalization"]["zero_sum_pairs"] == 1
    assert len(g.splits) == 3


def test_wikics_loader_on_fixture(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "wikics"
    d.mkdir(parents=True)
    n = 4
    data = {"features": [[0.0]] * n, "labels": [0, 1, 0, 1], "links": [[1], [0, 2], [3], []],
            "train_masks": [[True, True, False, False]] * 20, "val_masks": [[False, False, True, False]] * 20,
            "test_mask": [False, False, False, True]}
    (d / "data.json").write_text(json.dumps(data))
    g = load("wikics")
    assert len(g.splits) == 20 and g.edges.tolist() == [[0, 1], [1, 2], [2, 3]]
