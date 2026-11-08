"""PR-2 tests: canonicalisation, statistics, label isolation, loaders on fixtures.

No network access: loader tests write fixture files into a temporary
EDSP_DATA_DIR, which acquisition.fetch treats as a cache hit.
"""

import json
import math
import pickle
import re
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


# ------------------------------------------------------------- PR-3 loaders
def _write_fraud_zip(path, n, labels, rels):
    import io

    from scipy.io import savemat

    buf = io.BytesIO()
    savemat(buf, {"label": np.asarray(labels).reshape(1, -1), "features": sp.csr_matrix(np.ones((n, 2))),
                  **{k: sp.csr_matrix(v) for k, v in rels.items()}})
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.name.replace(".zip", ".mat"), buf.getvalue())


def _adj(n, pairs):
    A = np.zeros((n, n))
    for a, b in pairs:
        A[a, b] = A[b, a] = 1.0
    return A


def test_fraud_loader_relation_bitmask_on_fixture(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "fraud"
    d.mkdir(parents=True)
    n = 10
    rels = {"net_rur": _adj(n, [(0, 1), (2, 3)]), "net_rtr": _adj(n, [(0, 1), (4, 5)]),
            "net_rsr": _adj(n, [(0, 1), (2, 3), (6, 7)]) + np.eye(n)}
    _write_fraud_zip(d / "YelpChi.zip", n, np.arange(n) % 2, rels)
    g = load("yelpchi")
    got = {tuple(e): int(r) for e, r in zip(g.edges.tolist(), g.edge_relation, strict=True)}
    assert got == {(0, 1): 0b111, (2, 3): 0b101, (4, 5): 0b010, (6, 7): 0b100}
    assert g.meta["canonicalization"]["relation_combination_counts"] == {2: 1, 4: 1, 5: 1, 7: 1}
    assert g.edge_sign is None and len(g.splits) == 10


def test_amazon_fraud_unlabelled_prefix_is_excluded(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "fraud"
    d.mkdir(parents=True)
    n = 3325
    labels = np.zeros(n, dtype=int)
    labels[-10:] = 1
    rels = {k: _adj(n, [(i, i + 1) for i in range(0, n - 1, 2)]) for k in ("net_upu", "net_usu", "net_uvu")}
    _write_fraud_zip(d / "Amazon.zip", n, labels, {k: sp.csr_matrix(v) for k, v in rels.items()})
    g = load("amazon-fraud")
    for s in g.splits:
        for idx in (s.train_idx, s.val_idx, s.test_idx):
            assert idx.min() >= 3305
    st = S.label_stats(g)
    assert st["class_counts"] == [10, 10]


def test_signed_wiki_editor_keeps_labelled_lcc_and_maps_signs(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "wiki-editor"
    d.mkdir(parents=True)
    # 10-11-12-13 labelled path (+,-,+); 13-99 to an unlabelled node; 20-21 a smaller labelled component
    (d / "wiki_edit.txt").write_text("10 11 1\n11 12 0\n12 13 1\n13 99 1\n20 21 1\n11 10 1\n")
    (d / "wiki_usr_labels.txt").write_text("10 1\n11 0\n12 1\n13 0\n20 1\n21 0\n")
    g = load("wiki-editor")
    assert g.num_nodes == 4 and g.num_classes == 2
    assert g.edges.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert g.edge_sign.tolist() == [2.0, -1.0, 1.0]  # duplicate (10,11) summed
    assert g.meta["canonicalization"]["lcc_nodes"] == 4


def test_signed_wiki_rfa_and_elec_parsers_on_fixture(tmp_path):
    import gzip

    from edsp.data.loaders import load

    d = _cache(tmp_path) / "wiki-rfa"
    d.mkdir(parents=True)
    blocks = [("A", "B", 1, 1), ("C", "B", -1, 1), ("B", "A", 1, -1), ("C", "A", 0, -1), ("A", "C", -1, 1)]
    text = "\n\n".join(f"SRC:{s}\nTGT:{t}\nVOT:{v}\nRES:{r}\nYEA:2010\nDAT:x\nTXT:y" for s, t, v, r in blocks)
    with gzip.open(d / "wiki-RfA.txt.gz", "wt") as fh:
        fh.write(text + "\n")
    g = load("wiki-rfa")
    assert g.num_nodes == 3 and sorted(g.edge_sign.tolist()) == [-1.0, -1.0, 2.0]
    assert g.meta["canonicalization"]["leakage_note"]

    d = _cache(tmp_path) / "wiki-elec"
    d.mkdir(parents=True)
    lines = ["# comment", "E\t1", "T\t2004", "U\t1\tu1", "N\t2\tu2", "V\t1\t2\tt\tu2", "V\t-1\t3\tt\tu3", "V\t0\t4\tt\tu4",
             "E\t0", "T\t2005", "U\t2\tu2", "N\t1\tu1", "V\t-1\t1\tt\tu1", "V\t1\t3\tt\tu3",
             "E\t1", "T\t2006", "U\t3\tu3", "N\t1\tu1", "V\t1\t1\tt\tu1"]
    with gzip.open(d / "wikiElec.ElecBs3.txt.gz", "wt", encoding="latin-1") as fh:
        fh.write("\n".join(lines) + "\n")
    g = load("wiki-elec")
    assert g.num_nodes == 3 and g.num_classes == 2
    assert g.edges.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert g.edge_sign.tolist() == [0.0, 0.0, 1.0]  # (1,2): +1-1; (1,3): -1+1; (2,3): +1


def test_homophily_ignores_unlabelled_nodes():
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    y = np.array([0, 0, -1, 1])
    assert S.edge_homophily(edges, y) == 1.0
    loc = S.local_homophily(edges, y, 4)
    assert loc[0] == 1.0 and math.isnan(loc[3])


def test_stratified_accuracy_and_bins():
    from edsp.eval.evaluate import stratified_accuracy

    n = 6
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1, 0, 0])
    g = GraphData("toy", n, edges, 2, [Split("s", np.array([0]), np.array([1]), np.arange(2, 6), "test")], y)
    bins = S.local_homophily_bins(g, n_bins=2)
    # local homophily: 1, .5, .5, .5, .5, 1 -> bins 1,1,1,1,1,1 (0.5*2 = 1)
    assert bins.tolist() == [1, 1, 1, 1, 1, 1]
    assert S.local_homophily_bins(g).tolist() == [9, 5, 5, 5, 5, 9]
    assert S.degree_bins(g).tolist() == [1, 2, 2, 2, 2, 1]
    probs = np.tile([1.0, 0.0], (n, 1))
    out = stratified_accuracy(g.vault(0), probs, bins, n_bins=2)
    assert sum(out["count"]) == 4 and sum(out["correct"]) == 2


def _write_xlsx(path, sheets):
    """Tiny XLSX writer for fixtures (inline strings and numbers only)."""
    def col(i):
        s = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rns}><sheets>' + "".join(
            f'<sheet name="{n}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, n in enumerate(sheets)) + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(
            f'<Relationship Id="rId{i + 1}" Target="worksheets/sheet{i + 1}.xml" Type="x"/>' for i in range(len(sheets))) + "</Relationships>")
        for i, rows in enumerate(sheets.values()):
            xml = []
            for r, row in enumerate(rows):
                cells = []
                for c, v in enumerate(row):
                    ref = f"{col(c)}{r + 1}"
                    cells.append(f'<c r="{ref}"><v>{v}</v></c>' if isinstance(v, (int, float)) else
                                 f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>')
                xml.append(f'<row r="{r + 1}">' + "".join(cells) + "</row>")
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", f"<worksheet {ns}><sheetData>" + "".join(xml) + "</sheetData></worksheet>")


def test_congress_loader_parses_signed_matrix_and_party(tmp_path):
    import json as _json

    from edsp.data.loaders import load

    d = _cache(tmp_path) / "congress"
    d.mkdir(parents=True)
    names = ["Adams (D-NY)", "Baker (R-TX)", "Clark (D-CA)", "Dunn (I-VT)", "Evans (R-OH)"]
    M = np.array([[0, -1, 1, 0, -1], [-1, 0, -1, 0, 1], [1, -1, 0, 1, 0], [0, 0, 1, 0, 0], [-1, 1, 0, 0, 0]])
    _write_xlsx(d / "a_sign_of_the_times.xlsx", {"H92": [["x"], [0]], "S100": [names] + M.tolist()})
    (d / "figshare-8096429.json").write_text(_json.dumps({"files": [{"name": "a_sign_of_the_times.xlsx", "download_url": "https://example.invalid/x"}]}))
    g = load("congress-senate-100")
    assert g.num_nodes == 5 and g.num_classes == 2
    assert g._labels.tolist() == [0, 1, 0, -1, 1]
    got = {tuple(e): s for e, s in zip(g.edges.tolist(), g.edge_sign.tolist(), strict=True)}
    assert got == {(0, 1): -1.0, (0, 2): 1.0, (0, 4): -1.0, (1, 2): -1.0, (1, 4): 1.0, (2, 3): 1.0}
    assert all(3 not in s.train_idx and 3 not in s.test_idx for s in g.splits)
    # the layout of the real figshare file: no header row, column A = "Name (ST-P)"
    rl = ["Adams, A. (NY-D)", "Baker, B. (TX-R)", "Clark, C. (CA-D)", "Dunn, D. (VT-I)", "Evans, E. (OH-R)"]
    (d / "a_sign_of_the_times.xlsx").unlink()
    _write_xlsx(d / "a_sign_of_the_times.xlsx", {"S100": [[a] + r for a, r in zip(rl, M.tolist(), strict=True)]})
    g2 = load("congress-senate-100")
    assert g2._labels.tolist() == [0, 1, 0, -1, 1] and g2.edges.tolist() == g.edges.tolist()


def test_party_parser_formats():
    from edsp.data.loaders import _party_of

    assert _party_of("Baucus, M. (MT-D)") == "D" and _party_of("Jeffords, J. (VT-I)") == "I"
    assert [_party_of(x) for x in ["Smith (D-NY)", "SMITH_R_TX", "Jones [I-VT]", "Lee, Democrat", "nobody"]] == ["D", "R", "I", "D", None]


# ------------------------------------------------ PR-20 native-mixed extension loaders
NATIVE_MIXED = {"fb100-penn94", "fb100-amherst41", "fb100-cornell5", "fb100-johnshopkins55", "fb100-reed98",
                "genius", "deezer-europe", "twitch-de", "twitch-en", "twitch-es", "twitch-fr", "twitch-pt",
                "twitch-ru", "actor"}


def _check_canonical_and_isolated(g, fractions=(0.6, 0.2, 0.2), n_splits=10, random=True):
    """Undirected simple graph, splits over labelled nodes only (stratified when random),
    and the label-isolation contract of LabelVault."""
    from edsp.baselines.propagation import adjacency

    e = g.edges
    assert (e[:, 0] < e[:, 1]).all()  # one row per undirected edge, no self-loops
    assert len({tuple(x) for x in e.tolist()}) == len(e)
    A = adjacency(g)
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0
    y = g._labels  # tests may inspect the raw labels; methods never do
    labelled = np.flatnonzero(y >= 0)
    assert len(g.splits) == n_splits
    for s in g.splits:
        allidx = np.concatenate([s.train_idx, s.val_idx, s.test_idx])
        assert len(allidx) == len(np.unique(allidx))
        if random:
            assert sorted(allidx.tolist()) == labelled.tolist()
            for c in np.unique(y[labelled]):
                n_c = int(np.sum(y == c))
                a, b = round(fractions[0] * n_c), round(fractions[1] * n_c)
                assert [int(np.sum(y[idx] == c)) for idx in (s.train_idx, s.val_idx, s.test_idx)] == [a, b, n_c - a - b]
    v = g.vault(0)
    with pytest.raises(LabelAccessError):
        v.test_labels()
    with pytest.raises(LabelAccessError):
        v.val_labels()
    with pytest.raises(LabelAccessError):
        g.descriptive_vault().descriptive_labels()
    idx, ytr = v.train_labels()
    assert (ytr >= 0).all() and idx.tolist() == g.splits[0].train_idx.tolist()
    assert g.meta["extension"] == "native-mixed"


def test_linkx_label_binarize_encoding():
    from edsp.data.loaders import _label_binarize_columns

    cols = np.array([[1, 0, 9, 4], [2, 0, 9, 5], [3, 0, 8, 4], [1, 0, 8, 6]])
    X, widths = _label_binarize_columns(cols)
    # >= 3 values: one-hot; 1 value: one zero column; 2 values: indicator of the larger value
    assert widths == [3, 1, 1, 3]
    assert X.toarray().tolist() == [[1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]]


def test_fb100_loader_gender_labels_missing_and_features(tmp_path):
    from scipy.io import savemat

    from edsp.data.loaders import load

    d = _cache(tmp_path) / "linkx" / "facebook100"
    d.mkdir(parents=True)
    n = 6
    A = sp.csc_matrix(_adj(n, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]) + np.diag([0, 0, 1, 0, 0, 0]))
    info = np.array([  # status, gender (0 = missing), major, minor, dorm, year, high school
        [1, 1, 0, 0, 3, 2005, 1],
        [1, 2, 5, 0, 3, 2005, 2],
        [2, 0, 5, 0, 4, 2005, 3],
        [2, 2, 7, 0, 4, 2006, 4],
        [1, 1, 7, 0, 3, 2006, 5],
        [6, 2, 9, 0, 4, 2006, 6],
    ])
    savemat(d / "Penn94.mat", {"A": A, "local_info": info.astype(float)})
    g = load("fb100-penn94")
    assert g.num_nodes == 6 and g.num_classes == 2
    assert g._labels.tolist() == [0, 1, -1, 1, 0, 1]
    assert g.edges.tolist() == [[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]
    rep = g.meta["canonicalization"]
    assert rep["self_loops_removed"] == 1 and rep["duplicates_merged"] == 6 and rep["unlabelled_nodes"] == 1
    assert rep["feature_widths"] == [3, 4, 1, 1, 1, 6]
    assert g.features.shape == (6, 16) and g.features.dtype == np.float32
    # per row: one-hot status, major, high school + zero minor column + dorm / year indicators
    assert g.features.sum(axis=1).tolist() == [3.0, 3.0, 4.0, 5.0, 4.0, 5.0]
    _check_canonical_and_isolated(g)
    assert all(2 not in np.concatenate([s.train_idx, s.val_idx, s.test_idx]) for s in g.splits)
    st = S.label_stats(g)
    assert st["class_counts"] == [2, 3] and st["edge_homophily"] == 0.0


def test_fb100_url_encodes_the_space_in_johns_hopkins(monkeypatch):
    from edsp.data import acquisition
    from edsp.data.loaders import load

    seen = []

    def fake_fetch(url, relpath, **kw):
        seen.append((url, relpath))
        raise acquisition.AcquisitionError("offline")

    monkeypatch.setattr(acquisition, "fetch", fake_fetch)
    with pytest.raises(acquisition.AcquisitionError):
        load("fb100-johnshopkins55")
    url, relpath = seen[0]
    assert url.endswith("/data/facebook100/Johns%20Hopkins55.mat") and " " not in url
    assert relpath == "linkx/facebook100/JohnsHopkins55.mat"


def test_genius_loader_edge_index_on_fixture(tmp_path):
    from scipy.io import savemat

    from edsp.data.loaders import load

    d = _cache(tmp_path) / "linkx"
    d.mkdir(parents=True)
    ei = np.array([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9], [1, 0, 2, 2, 4, 3, 0, 6, 7, 8, 9, 5]], dtype=np.int64)
    y = np.array([[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]])
    savemat(d / "genius.mat", {"edge_index": ei, "node_feat": np.arange(120, dtype=float).reshape(10, 12), "label": y})
    g = load("genius")
    assert g.num_nodes == 10 and g.num_classes == 2 and g.features.shape == (10, 12)
    assert g.edges.tolist()[:4] == [[0, 1], [0, 4], [1, 2], [3, 4]] and len(g.edges) == 9
    assert g.meta["canonicalization"]["self_loops_removed"] == 1
    assert g.meta["canonicalization"]["label_counts"] == {0: 6, 1: 4}
    _check_canonical_and_isolated(g)


def test_deezer_loader_and_dense_feature_budget(tmp_path, monkeypatch):
    from scipy.io import savemat

    from edsp.data import loaders
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "linkx"
    d.mkdir(parents=True)
    n = 6
    A = sp.csc_matrix(_adj(n, [(0, 1), (2, 3), (4, 5), (1, 4)]) + np.diag([0, 0, 0, 1, 0, 0]))
    feats = sp.csc_matrix(np.eye(n, 8))
    savemat(d / "deezer-europe.mat", {"A": A, "label": np.array([[0, 1, 1, 0, 1, 0]]), "features": feats})
    g = load("deezer-europe")
    assert g.num_nodes == 6 and g.features.shape == (6, 8) and g._labels.tolist() == [0, 1, 1, 0, 1, 0]
    assert g.edges.tolist() == [[0, 1], [1, 4], [2, 3], [4, 5]]
    _check_canonical_and_isolated(g)
    monkeypatch.setattr(loaders, "MAX_DENSE_FEATURE_BYTES", 16)
    g2 = load("deezer-europe")
    assert g2.features is None
    assert g2.meta["canonicalization"]["features_omitted"]["shape"] == [6, 8]
    assert g2.edges.tolist() == g.edges.tolist()


def test_twitch_loader_on_fixture_with_duplicate_rows(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "linkx" / "twitch" / "FR"
    d.mkdir(parents=True)
    (d / "musae_FR_target.csv").write_text(
        "id,days,mature,views,partner,new_id\n"
        "101,10,True,5,False,2\n100,20,False,7,False,0\n102,30,True,1,False,1\n"
        "103,40,False,2,False,3\n104,50,True,3,False,4\n105,60,False,3,False,5\n"
        "106,60,False,3,False,6\n102,30,True,1,False,1\n")
    (d / "musae_FR_edges.csv").write_text("from,to\n0,1\n1,2\n2,0\n3,4\n4,3\n2,2\n5,6\n4,6\n")
    (d / "musae_FR_features.json").write_text(json.dumps({"0": [0, 3], "1": [3], "2": [5], "3": [], "4": [0],
                                                          "5": [3, 3], "6": [0], "9": [9]}))
    g = load("twitch-fr")
    assert g.num_nodes == 7 and g.num_classes == 2
    assert g._labels.tolist() == [0, 1, 1, 0, 1, 0, 0]
    assert g.edges.tolist() == [[0, 1], [0, 2], [1, 2], [3, 4], [4, 6], [5, 6]]
    rep = g.meta["canonicalization"]
    assert rep["duplicate_target_rows"] == 1 and rep["conflicting_duplicate_labels"] == 0
    assert rep["feature_keys_out_of_range"] == 1 and rep["feature_columns_kept"] == 3
    assert g.features.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
    _check_canonical_and_isolated(g)


def test_actor_loader_uses_the_fixed_geom_gcn_splits(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "geom-gcn"
    (d / "film").mkdir(parents=True)
    (d / "splits").mkdir(parents=True)
    (d / "film" / "out1_node_feature_label.txt").write_text(
        "node_id\tfeature\tlabel\n3\t1,4\t2\n0\t0\t0\n1\t2,3\t1\n2\t931\t4\n4\t5\t3\n5\t0,1\t0\n")
    (d / "film" / "out1_graph_edges.txt").write_text("node_id\tnode_id\n0\t1\n1\t0\n1\t2\n3\t3\n4\t5\n2\t5\n")
    base = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=np.uint8)
    for i in range(10):
        tr, va, te = np.roll(base, i, axis=1)
        np.savez(d / "splits" / f"film_split_0.6_0.2_{i}.npz", train_mask=tr, val_mask=va, test_mask=te)
    g = load("actor")
    assert g.num_nodes == 6 and g.num_classes == 5
    assert g._labels.tolist() == [0, 1, 4, 2, 3, 0]
    assert g.features.shape == (6, 932) and g.features[2, 931] == 1 and g.features.sum() == 9
    assert g.edges.tolist() == [[0, 1], [1, 2], [2, 5], [4, 5]]
    assert g.splits[0].test_idx.tolist() == [4, 5] and g.splits[1].test_idx.tolist() == [0, 5]
    assert g.meta["canonicalization"]["split_sizes"] == [[3, 1, 2]] * 10
    _check_canonical_and_isolated(g, random=False)
    np.savez(d / "splits" / "film_split_0.6_0.2_9.npz", train_mask=base[0, :5], val_mask=base[1, :5], test_mask=base[2, :5])
    with pytest.raises(ValueError, match="length 5"):
        load("actor")


def test_native_mixed_registry_entries():
    from edsp.data.loaders import LOADERS

    ds = registry()["datasets"]
    ext = {k for k, v in ds.items() if v.get("extension")}
    assert ext == NATIVE_MIXED and ext <= set(LOADERS)
    assert "native-mixed" in registry()["extensions"]
    for k in ext:
        v = ds[k]
        assert v["extension"] == "native-mixed" and v["status"] == "registered"
        assert v["metric"] in ("accuracy", "roc_auc") and v["cell"] == "C2"
        assert "TO VERIFY" in v["notes"] and "TO VERIFY" in v["license"]
        for r in v["routes"]:  # pinned to a commit, never a moving branch
            assert re.search(r"/[0-9a-f]{40}/", r) and r.startswith("https://raw.githubusercontent.com/")
    assert ds["penn94"]["loader"] is None and "fb100-penn94" in ds["penn94"]["status"]


def test_e1_default_lists_exclude_extension_entries():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / ".github" / "workflows" / "e1.yml").read_text()
    line = next(x for x in text.splitlines() if 'json.load(open("data/REGISTRY.json"))' in x)
    code = re.search(r"python3 -c '(.+?)' \"\$\(", line).group(1)
    ds = registry()["datasets"]
    for suite in ("E1", "E2"):
        out = subprocess.run([sys.executable, "-c", code, suite], cwd=root, capture_output=True, text=True, check=True)
        names = json.loads(out.stdout)
        assert names and not set(names) & NATIVE_MIXED
        assert all(not ds[k].get("extension") for k in names)
        assert ("cora" in names) == (suite == "E1") and ("wiki-editor" in names) == (suite == "E2")
