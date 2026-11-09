"""A18 extension loaders (GraphLand, OGB arxiv/products, LINKX arxiv-year,
twitch-gamers, snap-patents, pokec) on tiny synthetic files in the real formats.

No network access: fixtures are written into a temporary EDSP_DATA_DIR, which
acquisition.fetch treats as a cache hit.
"""

import gzip
import io
import json
import pickle
import zipfile

import numpy as np
import pytest
import scipy.sparse as sp

from edsp.data import acquisition
from edsp.data.registry import datasets_with_loader, default_datasets, registry

A18 = {"hm-categories", "artnet-exp", "city-reviews", "web-fraud", "web-topics", "ogbn-arxiv", "ogbn-products",
       "arxiv-year", "twitch-gamers", "snap-patents", "pokec"}


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("EDSP_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EDSP_ACQUISITION_LOG", str(tmp_path / "acq.jsonl"))
    yield


def _cache(tmp_path):
    return tmp_path / "cache"


def _canonical(g):
    e = g.edges
    assert (e[:, 0] < e[:, 1]).all() and len(np.unique(e, axis=0)) == len(e)


# ------------------------------------------------------------------ registry
def test_a18_registry_entries_are_opt_in_and_out_of_default_lists():
    ds = registry()["datasets"]
    a18 = {k for k, v in ds.items() if v.get("extension") == "a18"}
    assert a18 == A18 and A18 <= set(datasets_with_loader())
    assert not A18 & set(default_datasets())
    for k in A18:
        v = ds[k]
        assert v["opt_in"] is True and "TO VERIFY" in v["notes"] and v["routes"]
        assert v["metric"] in ("accuracy", "average_precision") and "expected_reported" in v
    for k in ("hm-categories", "artnet-exp", "city-reviews", "web-fraud", "web-topics"):
        assert len(ds[k]["md5"][f"{k}.zip"]) == 32


# ------------------------------------------------------------- acquisition
class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_checks_md5_and_rejects_html(tmp_path, monkeypatch):
    payload = b"PK\x03\x04 not really a zip"
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", lambda req, timeout: _FakeResponse(payload))
    import hashlib

    good = hashlib.md5(payload).hexdigest()
    with pytest.raises(acquisition.AcquisitionError, match="md5"):
        acquisition.fetch("https://x/a.zip", "t/a.zip", dataset="d", cell_id="C1", candidate_rank=1, expected_md5="0" * 32)
    assert not (_cache(tmp_path) / "t" / "a.zip").exists()
    path = acquisition.fetch("https://x/a.zip", "t/a.zip", dataset="d", cell_id="C1", candidate_rank=1, expected_md5=good)
    assert path.read_bytes() == payload
    monkeypatch.setattr(acquisition.urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(b"\n<!DOCTYPE html><html>quota exceeded</html>"))
    with pytest.raises(acquisition.AcquisitionError, match="HTML"):
        acquisition.fetch("https://x/b.mat", "t/b.mat", dataset="d", cell_id="C1", candidate_rank=1)
    log = [json.loads(x) for x in (tmp_path / "acq.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in log] == ["FAILED", "OK", "FAILED"]
    assert log[0]["failure_reason"] == "checksum" and log[2]["failure_reason"] == "format"
    assert log[1]["md5_raw"] == good and log[1]["md5_expected"] == good and log[1]["sha256_raw"].startswith("sha256:")


# ------------------------------------------------------------------ helpers
def test_simple_yaml_parser_without_pyyaml(monkeypatch):
    import builtins

    from edsp.data.loaders import _parse_simple_yaml

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "yaml" else real_import(name, *a, **k))
    text = ("dataset_name: artnet-exp\ntask: binary_classification\nnum_classes: 2\nhas_nans: true\n"
            "numerical_features_names:\n- views\n- 'share: of likes'\n- frac_a\nfraction_features_names:\n  - frac_a\n"
            "categorical_features_names: []\nother: [a, 'b c']\ntarget_name: \"is_explicit\"\n")
    d = _parse_simple_yaml(text)
    assert d["task"] == "binary_classification" and d["num_classes"] == 2 and d["has_nans"] is True
    assert d["numerical_features_names"] == ["views", "share: of likes", "frac_a"]
    assert d["fraction_features_names"] == ["frac_a"] and d["categorical_features_names"] == []
    assert d["other"] == ["a", "b c"] and d["target_name"] == "is_explicit"


def test_numeric_lines_handles_empty_fields_and_booleans():
    from edsp.data.loaders import _numeric_lines

    a = _numeric_lines(["0,,1.5,\n", ",True,False,2\r\n", "2,,,3\n"])
    assert a.shape == (3, 4)
    assert np.isnan(a[0, 1]) and np.isnan(a[0, 3]) and np.isnan(a[1, 0]) and np.isnan(a[2, 1]) and np.isnan(a[2, 2])
    assert a[1, 1:].tolist() == [1.0, 0.0, 2.0] and a[0, 2] == 1.5


def test_quantile_normal_properties_and_sklearn_parity():
    from edsp.data.loaders import _quantile_normal

    rng = np.random.default_rng(0)
    x = rng.lognormal(size=3000).astype(np.float32)
    x[:50] = np.nan
    x[50:300] = 1.0  # ties
    filled = np.where(np.isnan(x), np.float32(1.0), x)
    z = _quantile_normal(filled, x)
    assert np.isfinite(z).all() and abs(z.max()) < 5.3 and abs(z.min()) < 5.3
    order = np.argsort(filled, kind="stable")
    assert (np.diff(z[order]) >= -1e-12).all()  # monotone
    sk = pytest.importorskip("sklearn.preprocessing")
    qt = sk.QuantileTransformer(output_distribution="normal", subsample=None, random_state=0, copy=True).fit(x[:, None])
    np.testing.assert_allclose(z, qt.transform(filled[:, None]).ravel(), rtol=0, atol=1e-6)


def test_even_quantile_labels_matches_linkx_definition():
    from edsp.data.loaders import even_quantile_labels

    years = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999])
    lab = even_quantile_labels(years, 5)
    assert lab.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    y2 = np.array([2000.0, 2000.0, 2000.0, 2001.0, np.nan])
    assert even_quantile_labels(y2, 2).tolist() == [1, 1, 1, 1, -1]  # quantile = 2000 -> all >= upper


def _npy_splits(path, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(splits), dtype=object)
    for i, s in enumerate(splits):
        arr[i] = s
    np.save(path, arr, allow_pickle=True)


def test_split_unpickler_refuses_foreign_globals(tmp_path):
    from edsp.data.loaders import _linkx_fixed_splits

    p = tmp_path / "evil.npy"
    arr = np.empty(1, dtype=object)
    arr[0] = {"train": __import__("os").getcwd}  # a global outside numpy/torch
    np.save(p, arr, allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        _linkx_fixed_splits(p, np.zeros(4, dtype=np.int64), "t")


# ---------------------------------------------------------------- GraphLand
def _graphland_zip(path, name, *, n=8, directed=False, with_th=True):
    info = (f"dataset_name: {name}\ntask: binary_classification\ntarget_name: target\ngraph_is_directed: {str(directed).lower()}\n"
            "numerical_features_names:\n- views\n- frac_a\nfraction_features_names:\n- frac_a\n"
            "categorical_features_names:\n- city\n- flag\n")
    views = ["1", "", "3", "4", "5", "6", "7", "2"]
    frac = ["0.5", "0.25", "", "0.25", "0.1", "0.2", "0.3", "0.4"]
    city = ["1", "2", "3", "1", "2", "3", "1", "2"]
    flag = ["0", "1", "0", "1", "0", "1", "0", "1"]
    feats = "node_id,city,views,frac_a,flag\n" + "".join(f"{i},{city[i]},{views[i]},{frac[i]},{flag[i]}\n" for i in range(n))
    target = ["0.0", "1.0", "0.0", "1.0", "", "0.0", "1.0", "0.0"]
    targets = "node_id,target\n" + "".join(f"{i},{target[i]}\n" for i in range(n))
    edges = "source,target,weight\n0,1,1.0\n1,0,1.0\n1,2,2.0\n2,2,1.0\n3,4,1.0\n5,6,1.0\n6,7,1.0\n0,1,3.0\n"
    rh = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
    rl = ["test"] * 6 + ["train", "val"]

    def masks(assign):
        return "node_id,train,val,test\n" + "".join(
            f"{i},{a == 'train'},{a == 'val'},{a == 'test'}\n" for i, a in enumerate(assign))

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{name}/info.yaml", info)
        zf.writestr(f"{name}/features.csv", feats)
        zf.writestr(f"{name}/targets.csv", targets)
        zf.writestr(f"{name}/edgelist.csv", edges)
        zf.writestr(f"{name}/split_masks_RH.csv", masks(rh))
        zf.writestr(f"{name}/split_masks_RL.csv", masks(rl).replace("True", "1").replace("False", "0"))
        if with_th:
            zf.writestr(f"{name}/split_masks_TH.csv", masks(rh[::-1]))


def test_graphland_loader_on_fixture(tmp_path, monkeypatch):
    from edsp.data import loaders
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "graphland"
    d.mkdir(parents=True)
    _graphland_zip(d / "artnet-exp.zip", "artnet-exp")
    g = load("artnet-exp")
    assert g.num_nodes == 8 and g.num_classes == 2
    assert g._labels.tolist() == [0, 1, 0, 1, -1, 0, 1, 0]
    assert g.edges.tolist() == [[0, 1], [1, 2], [3, 4], [5, 6], [6, 7]]
    _canonical(g)
    rep = g.meta["canonicalization"]
    assert rep["self_loops_removed"] == 1 and rep["unlabelled_nodes"] == 1
    assert rep["edgelist_header"] == ["source", "target", "weight"]
    # features: [views (quantile-normal, NaN -> most frequent), frac_a (NaN -> most frequent 0.25),
    #            city one-hot (3 values), flag (binary -> one column)]
    X = g.features
    assert X.shape == (8, 1 + 1 + 3 + 1) and X.dtype == np.float32
    assert rep["features"]["dimension"] == 6 and rep["features"]["nan_counts"] == {"views": 1, "frac_a": 1}
    assert X[2, 1] == np.float32(0.25) and X[0, 1] == np.float32(0.5)
    assert X[:, 2:5].tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert X[:, 5].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    views = np.array([1, 1, 3, 4, 5, 6, 7, 2], dtype=float)  # node 1: NaN -> mode (1, the smallest of the ties)
    assert (np.argsort(X[:, 0], kind="stable") == np.argsort(views, kind="stable")).all()
    ids = [s.split_id for s in g.splits]
    assert ids == ["RH", "RL", "TH"] + [f"rand{i}" for i in range(10)]
    rh = g.splits[0]
    assert rh.train_idx.tolist() == [0, 1, 2, 3] and rh.val_idx.tolist() == [5] and rh.test_idx.tolist() == [6, 7]
    assert rep["provided_splits"]["RH"]["unlabelled_in_masks"] == 1
    assert g.splits[1].train_idx.tolist() == [6] and g.splits[1].val_idx.tolist() == [7]
    for s in g.splits:
        g.vault(s.split_id, strict=False)  # disjoint, labelled-only
        assert 4 not in np.concatenate([s.train_idx, s.val_idx, s.test_idx])
    assert g.meta["extension"] == "a18" and g.meta["symmetrised_from_directed"] is False
    # dense-feature budget: same graph, features omitted, dimension still reported
    monkeypatch.setattr(loaders, "MAX_DENSE_FEATURE_BYTES", 64)
    g2 = load("artnet-exp")
    assert g2.features is None and g2.meta["canonicalization"]["features_omitted"]["shape"] == [8, 6]
    assert g2.edges.tolist() == g.edges.tolist()


def test_graphland_city_reviews_without_temporal_split_and_web_is_directed(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "graphland"
    d.mkdir(parents=True)
    _graphland_zip(d / "city-reviews.zip", "city-reviews", with_th=False)
    assert [s.split_id for s in load("city-reviews").splits][:3] == ["RH", "RL", "rand0"]
    _graphland_zip(d / "web-fraud.zip", "web-fraud", directed=True)
    g = load("web-fraud")
    assert g.meta["symmetrised_from_directed"] is True and g.meta["canonicalization"]["duplicates_merged"] == 2


def test_graphland_rejects_missing_feature_columns(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "graphland"
    d.mkdir(parents=True)
    _graphland_zip(d / "hm-categories.zip", "hm-categories")
    with pytest.raises(ValueError, match="task"):  # fixture says binary, hm-categories is multiclass
        load("hm-categories")


# ---------------------------------------------------------------------- OGB
def _gz_csv(rows):
    return gzip.compress("".join(",".join(str(x) for x in r) + "\n" for r in rows).encode())


def _ogb_zip(path, root, split_dir, *, n=6, with_year=False):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{root}/raw/edge.csv.gz", _gz_csv([[0, 1], [1, 0], [1, 2], [3, 3], [4, 5], [2, 5]]))
        zf.writestr(f"{root}/raw/node-feat.csv.gz", _gz_csv([[i * 0.5, -i, 1.0] for i in range(n)]))
        zf.writestr(f"{root}/raw/node-label.csv.gz", _gz_csv([[c] for c in [0, 2, 1, 2, 0, 1]]))
        zf.writestr(f"{root}/raw/num-node-list.csv.gz", _gz_csv([[n]]))
        zf.writestr(f"{root}/raw/num-edge-list.csv.gz", _gz_csv([[6]]))
        if with_year:
            zf.writestr(f"{root}/raw/node_year.csv.gz", _gz_csv([[y] for y in [2001, 2005, 2010, 2015, 2018, 2020]]))
        for part, idx in (("train", [0, 1, 2]), ("valid", [3]), ("test", [5, 4])):
            zf.writestr(f"{root}/split/{split_dir}/{part}.csv.gz", _gz_csv([[i] for i in idx]))


def test_ogbn_arxiv_and_products_loaders_on_fixture(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path) / "ogb"
    d.mkdir(parents=True)
    _ogb_zip(d / "arxiv.zip", "arxiv", "time")
    _ogb_zip(d / "products.zip", "products", "sales_ranking")
    g = load("ogbn-arxiv")
    assert g.num_nodes == 6 and g.num_classes == 3 and g.features.shape == (6, 3) and g.features.dtype == np.float32
    assert g.edges.tolist() == [[0, 1], [1, 2], [2, 5], [4, 5]]
    _canonical(g)
    assert g.splits[0].split_id == "time" and g.splits[0].test_idx.tolist() == [4, 5]
    assert [s.split_id for s in g.splits[1:]] == [f"rand{i}" for i in range(10)]
    assert g.meta["symmetrised_from_directed"] is True and g.meta["canonicalization"]["raw_edge_rows"] == 6
    p = load("ogbn-products")
    assert p.splits[0].split_id == "sales_ranking" and p.meta["symmetrised_from_directed"] is False


def test_arxiv_year_uses_year_quantiles_and_linkx_fixed_splits(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path)
    (d / "ogb").mkdir(parents=True)
    _ogb_zip(d / "ogb" / "arxiv.zip", "arxiv", "time", with_year=True)
    _npy_splits(d / "linkx" / "splits" / "arxiv-year-splits.npy",
                [{"train": np.array([0, 1, 2]), "valid": np.array([3]), "test": np.array([4, 5])},
                 {"train": np.array([5, 4, 3]), "valid": np.array([2]), "test": np.array([1, 0])}])
    g = load("arxiv-year")
    assert g.num_classes == 5 and g._labels.tolist() == [0, 1, 2, 3, 4, 4]  # quantiles 2005, 2010, 2015, 2018
    assert [s.split_id for s in g.splits] == ["linkx0", "linkx1"] + [f"rand{i}" for i in range(10)]
    assert g.splits[1].train_idx.tolist() == [3, 4, 5] and g.splits[1].test_idx.tolist() == [0, 1]
    assert g.meta["canonicalization"]["linkx_splits"]["sizes"] == [[3, 1, 2], [3, 1, 2]]


# ------------------------------------------------------------------- LINKX
def test_twitch_gamers_loader_on_snap_style_fixture(tmp_path):
    from edsp.data.loaders import load

    d = _cache(tmp_path)
    (d / "snap").mkdir(parents=True)
    feats = ("views,mature,life_time,created_at,updated_at,numeric_id,dead_account,language,affiliate\n"
             "10,1,100,2015-01-02,2018-01-01,1,0,EN,1\n"
             "20,0,200,2016-01-02,2018-02-01,0,0,DE,0\n"
             "30,1,300,2017-01-02,2018-03-01,2,1,EN,0\n"
             "40,0,400,2014-01-02,2018-04-01,3,0,FR,1\n")
    with zipfile.ZipFile(d / "snap" / "twitch_gamers.zip", "w") as zf:
        zf.writestr("large_twitch_features.csv", feats)
        zf.writestr("large_twitch_edges.csv", "numeric_id_1,numeric_id_2\n0,1\n1,2\n2,1\n3,3\n")
    _npy_splits(d / "linkx" / "splits" / "twitch-gamer-splits.npy",
                [{"train": np.array([0, 1]), "valid": np.array([2]), "test": np.array([3])}])
    g = load("twitch-gamers")
    assert g.num_nodes == 4 and g._labels.tolist() == [0, 1, 1, 0]
    assert g.edges.tolist() == [[0, 1], [1, 2]]
    rep = g.meta["canonicalization"]
    assert rep["feature_columns"] == ["views", "life_time", "created_at", "updated_at", "dead_account", "language", "affiliate"]
    assert rep["rows_sorted_by_numeric_id"] is False and rep["languages"] == 3
    raw = np.array([[20, 200, 20160102, 20180201, 0, 1, 0], [10, 100, 20150102, 20180101, 0, 0, 1],
                    [30, 300, 20170102, 20180301, 1, 0, 0], [40, 400, 20140102, 20180401, 0, 2, 1]], dtype=float)
    expect = (raw - raw.mean(0)) / raw.std(0, ddof=1)  # language codes by first appearance in file order: EN 0, DE 1, FR 2
    np.testing.assert_allclose(g.features, expect, rtol=1e-6)
    assert [s.split_id for s in g.splits][:2] == ["linkx0", "rand0"]


def test_snap_patents_and_pokec_mat_loaders(tmp_path):
    from scipy.io import savemat

    from edsp.data.loaders import load

    d = _cache(tmp_path) / "linkx"
    d.mkdir(parents=True)
    ei = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0]], dtype=np.int64)
    years = np.arange(1990, 2000, dtype=float).reshape(-1, 1)
    savemat(d / "snap_patents.mat", {"edge_index": ei, "node_feat": sp.csr_matrix(np.eye(10, 4)), "num_nodes": 10, "years": years})
    g = load("snap-patents")
    assert g.num_nodes == 10 and g.num_classes == 5 and g._labels.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert len(g.edges) == 10 and g.features.shape == (10, 4) and g.meta["canonicalization"]["self_loops_removed"] == 1
    assert "not registered" in g.meta["canonicalization"]["linkx_splits"] and g.splits[0].split_id == "rand0"
    lab = np.array([[0, 1, -1, 1, 0, 1, 0, 1, 0, 1]])
    savemat(d / "pokec.mat", {"edge_index": ei, "node_feat": np.ones((10, 3)), "num_nodes": 10, "label": lab})
    p = load("pokec")
    assert p.num_classes == 2 and p.features.shape == (10, 3)
    for s in p.splits:
        assert 2 not in np.concatenate([s.train_idx, s.val_idx, s.test_idx])
