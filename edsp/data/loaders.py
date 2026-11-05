"""Dataset loaders for PR-2 (Master Prompt §3.1 cells C1, C2, C4.1, C5.4).

Every loader returns a :class:`GraphData` with canonical undirected edges
(symmetrised, self-loops removed, duplicates merged; see
``meta['canonicalization']``). Source URLs and their verification status live
in ``data/REGISTRY.json``; downloads go through :func:`acquisition.fetch`, which
logs every attempt. Only numpy/scipy/stdlib are used, so loaders run in the
locked environment without torch_geometric/ogb.
"""

from __future__ import annotations

import gzip
import io
import json
import pickle
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import acquisition
from .graph import GraphData, Split, canonicalize_edges, random_splits
from .registry import candidate_urls, cell_of

RAW = "https://raw.githubusercontent.com"


def _fetch(dataset: str, filename: str, relpath: str) -> Path:
    """Try each registered route for ``dataset``/``filename`` in rank order."""
    errors = []
    for rank, base in enumerate(candidate_urls(dataset), start=1):
        url = base.format(file=filename)
        try:
            return acquisition.fetch(url, relpath, dataset=dataset, cell_id=cell_of(dataset), candidate_rank=rank)
        except acquisition.AcquisitionError as exc:
            errors.append(str(exc))
    raise acquisition.AcquisitionError(f"all routes failed for {dataset}/{filename}: {errors}")


def _finish(g: GraphData, report: dict) -> GraphData:
    g.meta["canonicalization"] = report
    acquisition.log_processed(g.name, cell_of(g.name), g.content_hash(), report)
    return g


def _mask_splits(train: np.ndarray, val: np.ndarray, test: np.ndarray, source: str) -> list[Split]:
    train, val, test = (np.atleast_2d(m) for m in (train, val, test))
    return [
        Split(str(i), np.flatnonzero(train[i]), np.flatnonzero(val[i]), np.flatnonzero(test[i]), source)
        for i in range(train.shape[0])
    ]


# --------------------------------------------------------------------- Planetoid
class _LegacyUnpickler(pickle.Unpickler):
    """Map the 2016-era module paths in the Planetoid pickles to current scipy."""

    def find_class(self, module: str, name: str):
        if module.startswith("scipy.sparse"):
            return getattr(sp, name)
        return super().find_class(module, name)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return _LegacyUnpickler(f, encoding="latin1").load()


def planetoid(name: str) -> GraphData:
    """Cora / CiteSeer / PubMed from the original Planetoid files (Yang et al. 2016).

    Mirrors torch_geometric's ``read_planetoid_data``: test rows are re-ordered,
    CiteSeer's 15 isolated test-range nodes are padded with zero features and
    label 0 (they are not in any split), self-loops removed, edges deduplicated.
    """
    key = name.lower()
    parts = {}
    for suffix in ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"):
        fname = f"ind.{key}.{suffix}"
        parts[suffix] = _fetch(name, fname, f"planetoid/{fname}")
    tx, allx = (_load_pickle(parts[s]) for s in ("tx", "allx"))  # ind.*.x is the train subset of allx
    y, ty, ally = (np.asarray(_load_pickle(parts[s])) for s in ("y", "ty", "ally"))
    graph = _load_pickle(parts["graph"])
    test_index = np.array([int(line) for line in parts["test.index"].read_text().split()], dtype=np.int64)
    sorted_test = np.sort(test_index)
    tx = tx.toarray() if sp.issparse(tx) else np.asarray(tx)
    allx = allx.toarray() if sp.issparse(allx) else np.asarray(allx)
    padded = 0
    if key == "citeseer":
        span = sorted_test.max() - sorted_test.min() + 1
        tx_ext = np.zeros((span, tx.shape[1]), dtype=tx.dtype)
        ty_ext = np.zeros((span, ty.shape[1]), dtype=ty.dtype)
        tx_ext[sorted_test - sorted_test.min()] = tx
        ty_ext[sorted_test - sorted_test.min()] = ty
        padded = int(span - len(sorted_test))
        tx, ty = tx_ext, ty_ext
    features = np.vstack([allx, tx]).astype(np.float32)
    labels = np.vstack([ally, ty]).argmax(axis=1).astype(np.int64)
    features[test_index] = features[sorted_test].copy()
    labels[test_index] = labels[sorted_test].copy()
    n = len(labels)
    rows = [(int(u), int(v)) for u, nbrs in graph.items() for v in nbrs]
    edges, _, _, report = canonicalize_edges(np.array(rows, dtype=np.int64), n)
    report["citeseer_padded_nodes"] = padded
    official = Split(
        "public",
        np.arange(len(y)),
        np.arange(len(y), len(y) + 500),
        np.sort(test_index),
        "official Planetoid public split (Yang et al. 2016)",
    )
    k = int(labels.max() + 1)
    eligible = labels.copy()
    if padded:
        pad_nodes = np.setdiff1d(np.arange(sorted_test.min(), sorted_test.max() + 1), sorted_test)
        eligible[pad_nodes] = -1  # keep padding nodes out of random splits
    splits = [official] + random_splits(eligible, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, k, splits, labels, features=features,
                  meta={"domain": "citation", "source": "kimiyoung/planetoid"})
    return _finish(g, report)


# ------------------------------------------------------- Platonov et al. (2023)
def heterophilous_npz(name: str) -> GraphData:
    """roman-empire, amazon-ratings, minesweeper, tolokers, questions,
    chameleon-filtered, squirrel-filtered (yandex-research/heterophilous-graphs)."""
    fname = name.replace("-", "_") + ".npz"
    path = _fetch(name, fname, f"heterophilous/{fname}")
    d = np.load(path)
    labels = d["node_labels"].astype(np.int64)
    n = len(labels)
    edges, _, _, report = canonicalize_edges(d["edges"], n)
    splits = _mask_splits(d["train_masks"], d["val_masks"], d["test_masks"], "official (Platonov et al. 2023), 10 splits")
    g = GraphData(name, n, edges, int(labels.max() + 1), splits, labels,
                  features=d["node_features"].astype(np.float32),
                  meta={"source": "yandex-research/heterophilous-graphs"})
    return _finish(g, report)


# ------------------------------------------------------ Shchur et al. (2018) npz
def shchur_npz(name: str) -> GraphData:
    """Amazon-Computers/Photo, Coauthor-CS/Physics (shchur/gnn-benchmark npz).

    No official split exists; 10 stratified 60/20/20 splits (seeds 0-9) are used.
    The tunedGNN split convention is TO VERIFY (Master Prompt §3.1 C1.2).
    """
    files = {
        "amazon-computers": "amazon_electronics_computers.npz",
        "amazon-photo": "amazon_electronics_photo.npz",
        "coauthor-cs": "ms_academic_cs.npz",
        "coauthor-physics": "ms_academic_phy.npz",
    }
    fname = files[name]
    d = np.load(_fetch(name, fname, f"shchur/{fname}"), allow_pickle=True)
    n = int(d["adj_shape"][0])
    adj = sp.csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]), shape=tuple(d["adj_shape"])).tocoo()
    attr = sp.csr_matrix((d["attr_data"], d["attr_indices"], d["attr_indptr"]), shape=tuple(d["attr_shape"]))
    labels = d["labels"].astype(np.int64)
    edges, _, _, report = canonicalize_edges(np.stack([adj.row, adj.col], axis=1), n)
    g = GraphData(name, n, edges, int(labels.max() + 1), random_splits(labels, 10, (0.6, 0.2, 0.2), 0), labels,
                  features=attr.toarray().astype(np.float32), meta={"source": "shchur/gnn-benchmark"})
    return _finish(g, report)


# ------------------------------------------------------------------- WikiCS
def wikics(name: str = "wikics") -> GraphData:
    """WikiCS (Mernyei & Cangea 2020) with its 20 official train/val splits and single test mask."""
    path = _fetch(name, "data.json", "wikics/data.json")
    d = json.loads(path.read_text())
    labels = np.asarray(d["labels"], dtype=np.int64)
    n = len(labels)
    rows = [(i, j) for i, nbrs in enumerate(d["links"]) for j in nbrs]
    edges, _, _, report = canonicalize_edges(np.array(rows, dtype=np.int64), n)
    train = np.asarray(d["train_masks"], dtype=bool)
    val = np.asarray(d["val_masks"], dtype=bool)
    test = np.asarray(d["test_mask"], dtype=bool)
    splits = _mask_splits(train, val, np.tile(test, (train.shape[0], 1)), "official WikiCS, 20 splits")
    g = GraphData(name, n, edges, int(labels.max() + 1), splits, labels,
                  features=np.asarray(d["features"], dtype=np.float32), meta={"source": "pmernyei/wiki-cs-dataset"})
    return _finish(g, report)


# ------------------------------------------------------------------ Sampson
def parse_pajek_project(text: str) -> dict:
    """Minimal Pajek .paj parser: networks (vertices, arcs, edges) and partitions."""
    out: dict = {"networks": {}, "partitions": {}}
    current = None
    mode = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("*network"):
            current = {"names": [], "arcs": [], "edges": []}
            out["networks"][line.split(None, 1)[1].strip()] = current
            mode = None
        elif low.startswith("*partition"):
            current = {"values": []}
            out["partitions"][line.split(None, 1)[1].strip()] = current
            mode = "partition"
        elif low.startswith("*vertices"):
            mode = "values" if mode == "partition" else "vertices"
        elif low.startswith("*arcs"):
            mode = "arcs"
        elif low.startswith("*edges"):
            mode = "edges"
        elif current is not None and mode == "vertices":
            m = re.match(r'\s*\d+\s+"([^"]*)"', raw)
            current["names"].append(m.group(1) if m else line.split()[1])
        elif current is not None and mode in ("arcs", "edges"):
            tok = line.split()
            current[mode].append((int(tok[0]) - 1, int(tok[1]) - 1, float(tok[2]) if len(tok) > 2 else 1.0))
        elif current is not None and mode == "values":
            current["values"].append(int(line))
    return out


def sampson(name: str = "sampson") -> GraphData:
    """Sampson's monastery at T4 (18 novices) with the 4 factions of Sampson_factions_T4.

    Source: Pajek ESNA collection (de Nooy, Mrvar & Batagelj 2005), file
    Sampson.paj, network ``Sampson_T4``: directed signed arcs (+1 like, -1 dislike).
    Arcs are made undirected by summing signs (H-NATIVE rule, §2.3a); pairs whose
    reciprocal signs cancel to 0 are kept with sign 0 and counted. Case study only
    (excluded from rank statistics, Master Prompt §3.1 C5.4).
    """
    path = _fetch(name, "Sampson.zip", "sampson/Sampson.zip")
    text = zipfile.ZipFile(path).read("Sampson.paj").decode("latin1")
    proj = parse_pajek_project(text)
    net = proj["networks"]["Sampson_T4"]
    factions = np.asarray(proj["partitions"]["Sampson_factions_T4"]["values"], dtype=np.int64) - 1
    n = len(net["names"])
    arcs = np.array([(u, v) for u, v, _ in net["arcs"]], dtype=np.int64)
    signs = np.array([w for _, _, w in net["arcs"]], dtype=float)
    edges, _, sign, report = canonicalize_edges(arcs, n, edge_sign=signs)
    assert sign is not None
    report["zero_sum_pairs"] = int(np.sum(sign == 0))
    # Leave-one-out splits: each node is the single test node once; val is empty.
    splits = [
        Split(f"loo{i}", np.setdiff1d(np.arange(n), [i]), np.empty(0, dtype=np.int64), np.array([i]), "leave-one-out")
        for i in range(n)
    ]
    g = GraphData(name, n, edges, int(factions.max() + 1), splits, factions, edge_sign=sign,
                  meta={"source": "Pajek ESNA Sampson.paj (network Sampson_T4)", "names": net["names"],
                        "symmetrised_from_directed": True})
    return _finish(g, report)


# -------------------------------------------------------------- ogbn-proteins
def _read_csv_gz(zf: zipfile.ZipFile, member: str, dtype, ncols: int) -> np.ndarray:
    out = []
    with zf.open(member) as raw, gzip.open(raw) as gz:
        reader = io.TextIOWrapper(gz)
        while True:
            lines = reader.readlines(1 << 26)
            if not lines:
                break
            out.append(np.array(" ".join(lines).replace(",", " ").split(), dtype=dtype).reshape(-1, ncols))
    return np.concatenate(out) if out else np.empty((0, ncols), dtype=dtype)


def ogbn_proteins(name: str = "ogbn-proteins") -> GraphData:
    """ogbn-proteins from the raw OGB archive (no `ogb` dependency).

    Layout (TO VERIFY on first Actions run): proteins/raw/{edge,edge-feat,node-label,
    node_species}.csv.gz and proteins/split/species/{train,valid,test}.csv.gz.
    Edge features are the 8 STRING channel scores in [0, 1]; labels are 112 binary tasks.
    """
    path = _fetch(name, "proteins.zip", "ogb/proteins.zip")
    zf = zipfile.ZipFile(path)
    names = zf.namelist()

    def member(suffix: str) -> str:
        hits = [m for m in names if m.endswith(suffix)]
        if not hits:
            raise FileNotFoundError(f"{suffix} not in proteins.zip")
        return hits[0]

    raw_edges = _read_csv_gz(zf, member("raw/edge.csv.gz"), np.int64, 2)
    feats = _read_csv_gz(zf, member("raw/edge-feat.csv.gz"), np.float32, 8)
    labels = _read_csv_gz(zf, member("raw/node-label.csv.gz"), np.int8, 112)
    n = len(labels)
    edges, attr, _, report = canonicalize_edges(raw_edges, n, edge_attr=feats)
    split_idx = [_read_csv_gz(zf, member(f"split/species/{s}.csv.gz"), np.int64, 1).ravel() for s in ("train", "valid", "test")]
    g = GraphData(name, n, edges, 2, [Split("species", *split_idx, "official OGB species split")], labels,
                  edge_attr=attr, multilabel=True, meta={"source": "OGB proteins.zip", "tasks": 112})
    return _finish(g, report)


LOADERS: dict[str, Callable[[str], GraphData]] = {
    **{k: planetoid for k in ("cora", "citeseer", "pubmed")},
    **{k: heterophilous_npz for k in ("roman-empire", "amazon-ratings", "minesweeper", "tolokers",
                                       "questions", "chameleon-filtered", "squirrel-filtered")},
    **{k: shchur_npz for k in ("amazon-computers", "amazon-photo", "coauthor-cs", "coauthor-physics")},
    "wikics": wikics,
    "sampson": sampson,
    "ogbn-proteins": ogbn_proteins,
}


def load(name: str) -> GraphData:
    try:
        loader = LOADERS[name]
    except KeyError as exc:
        raise KeyError(f"no loader registered for {name!r}; known: {sorted(LOADERS)}") from exc
    return loader(name)
