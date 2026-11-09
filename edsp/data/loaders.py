"""Dataset loaders for PR-2 (Master Prompt §3.1 cells C1, C2, C4.1, C5.4).

Every loader returns a :class:`GraphData` with canonical undirected edges
(symmetrised, self-loops removed, duplicates merged; see
``meta['canonicalization']``). Source URLs and their verification status live
in ``data/REGISTRY.json``; downloads go through :func:`acquisition.fetch`, which
logs every attempt. Only numpy/scipy/stdlib are used, so loaders run in the
locked environment without torch_geometric/ogb.
"""

from __future__ import annotations

import csv
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
from .registry import candidate_urls, cell_of, dataset_entry

RAW = "https://raw.githubusercontent.com"


def _fetch(dataset: str, filename: str, relpath: str, *, expected_md5: str | None = None) -> Path:
    """Try each registered route for ``dataset``/``filename`` in rank order."""
    errors = []
    for rank, base in enumerate(candidate_urls(dataset), start=1):
        url = base.format(file=filename)
        try:
            return acquisition.fetch(url, relpath, dataset=dataset, cell_id=cell_of(dataset), candidate_rank=rank,
                                     expected_md5=expected_md5)
        except acquisition.AcquisitionError as exc:
            errors.append(str(exc))
    raise acquisition.AcquisitionError(f"all routes failed for {dataset}/{filename}: {errors}")


def _fetch_aux(dataset: str, key: str, filename: str, relpath: str) -> Path:
    """Like :func:`_fetch`, for a second source of ``dataset`` (registry ``aux_routes[key]``),
    e.g. fixed splits hosted elsewhere than the graph."""
    routes = dataset_entry(dataset).get("aux_routes", {}).get(key, [])
    if not routes:
        raise KeyError(f"{dataset}: no aux_routes[{key!r}] in data/REGISTRY.json")
    errors = []
    for rank, base in enumerate(routes, start=1):
        url = base.format(file=filename)
        try:
            return acquisition.fetch(url, relpath, dataset=dataset, cell_id=cell_of(dataset), candidate_rank=rank)
        except acquisition.AcquisitionError as exc:
            errors.append(str(exc))
    raise acquisition.AcquisitionError(f"all {key} routes failed for {dataset}/{filename}: {errors}")


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


# ------------------------------------------------- C3a: relation-typed fraud graphs
def fraud_mat(name: str) -> GraphData:
    """YelpChi and Amazon fraud graphs (Dou et al., CIKM 2020; CARE-GNN repository).

    Relations: YelpChi R-U-R, R-T-R, R-S-R; Amazon U-P-U, U-S-U, U-V-U. An undirected
    edge may belong to several relations, so ``edge_relation`` stores the bitmask of
    the relations containing it (bit r for relation r in the order above); H-RELATION
    then assigns one polarity per relation combination. Amazon: the first 3,305 nodes
    are unlabelled in the CARE-GNN data (TO VERIFY) and are excluded from all splits.
    Splits: 10 stratified random 40/20/40 splits (seeds 0-9), the convention of
    CARE-GNN / BWGNN (TO VERIFY).
    """
    from scipy.io import loadmat

    fname, rels = {"yelpchi": ("YelpChi.zip", ("net_rur", "net_rtr", "net_rsr")),
                   "amazon-fraud": ("Amazon.zip", ("net_upu", "net_usu", "net_uvu"))}[name]
    path = _fetch(name, fname, f"fraud/{fname}")
    member = fname.replace(".zip", ".mat")
    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        d = loadmat(io.BytesIO(fh.read()))
    labels = np.asarray(d["label"]).ravel().astype(np.int64)
    n = len(labels)
    rows, cols, bits = [], [], []
    for r, key in enumerate(rels):
        A = sp.coo_matrix(d[key])
        keep = A.row != A.col
        rows.append(A.row[keep])
        cols.append(A.col[keep])
        bits.append(np.full(int(keep.sum()), 1 << r, dtype=np.int64))
    raw = np.stack([np.concatenate(rows), np.concatenate(cols)], axis=1)
    edges, _, _mask, report = canonicalize_edges(raw, n, edge_sign=np.concatenate(bits).astype(float))
    # canonicalize sums the per-row values; recover the bitmask with a bitwise OR instead
    lo, hi = np.minimum(raw[:, 0], raw[:, 1]), np.maximum(raw[:, 0], raw[:, 1])
    key_raw = lo * n + hi
    key_out = edges[:, 0] * n + edges[:, 1]
    order = np.argsort(key_out)
    pos = order[np.searchsorted(key_out[order], key_raw)]
    relation = np.zeros(len(edges), dtype=np.int64)
    np.bitwise_or.at(relation, pos, np.concatenate(bits))
    report["relations"] = list(rels)
    report["relation_combination_counts"] = {int(b): int(c) for b, c in zip(*np.unique(relation, return_counts=True), strict=True)}
    if name == "amazon-fraud":
        labels[:3305] = -1  # unlabelled in the CARE-GNN data: excluded from splits and statistics
        report["unlabelled_prefix_nodes"] = 3305
    eligible = labels
    feats = sp.csr_matrix(d["features"]).toarray().astype(np.float32)
    g = GraphData(name, n, edges, 2, random_splits(eligible, 10, (0.4, 0.2, 0.4), 0), labels, features=feats,
                  edge_relation=relation, meta={"source": "YingtongDou/CARE-GNN data", "domain": "fraud"})
    return _finish(g, report)


# ------------------------------------------------------- C5: signed + node labels
def _signed_graph(name: str, nodes_raw: np.ndarray | None, src: np.ndarray, dst: np.ndarray, sign: np.ndarray,
                  label_of: dict, source: str, extra: dict | None = None) -> GraphData:
    """Mercado et al. (ECML-PKDD 2019) convention: keep labelled nodes only, then the
    largest connected component of the unsigned graph; signs summed per pair."""
    labelled = np.array(sorted(k for k in label_of), dtype=np.int64)
    idx = {int(u): i for i, u in enumerate(labelled)}
    keep = np.array([int(a) in idx and int(b) in idx and int(a) != int(b) for a, b in zip(src, dst, strict=True)], dtype=bool)
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in zip(src[keep], dst[keep], strict=True)], dtype=np.int64).reshape(-1, 2)
    n = len(labelled)
    edges, _, s, report = canonicalize_edges(e, n, edge_sign=sign[keep].astype(float))
    from scipy.sparse.csgraph import connected_components

    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    big = np.bincount(comp).argmax()
    sel = np.flatnonzero(comp == big)
    remap = -np.ones(n, dtype=np.int64)
    remap[sel] = np.arange(len(sel))
    m = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
    edges2 = np.stack([remap[edges[m, 0]], remap[edges[m, 1]]], axis=1)
    y_raw = np.array([label_of[int(u)] for u in labelled[sel]])
    classes = sorted(set(y_raw.tolist()))
    y = np.array([classes.index(v) for v in y_raw], dtype=np.int64)
    report |= {"labelled_nodes": int(n), "lcc_nodes": len(sel), "class_values": classes, **(extra or {})}
    g = GraphData(name, len(sel), edges2, len(classes), random_splits(y, 10, (0.6, 0.2, 0.2), 0), y,
                  edge_sign=s[m], meta={"source": source, "domain": "social/signed"})
    return _finish(g, report)


def wiki_editor(name: str = "wiki-editor") -> GraphData:
    """Wikipedia editor network (Yuan, Wu & Xiang, PAKDD 2017; files distributed with
    Mercado et al.'s code): signed edges (1 positive, 0 negative) between editors and
    binary user labels (meaning TO VERIFY: vandal vs benign in the SNE paper)."""
    e = _fetch(name, "wiki_edit.txt", "wiki-editor/wiki_edit.txt")
    lab = _fetch(name, "wiki_usr_labels.txt", "wiki-editor/wiki_usr_labels.txt")
    E = np.loadtxt(e, dtype=np.int64).reshape(-1, 3)
    L = np.loadtxt(lab, dtype=np.int64).reshape(-1, 2)
    label_of = {int(u): int(c) for u, c in L}
    sign = np.where(E[:, 2] == 1, 1.0, -1.0)
    return _signed_graph(name, None, E[:, 0], E[:, 1], sign, label_of, "melopeo/GL WikipediaDatasets/generated_wikiEditor_network")


def wiki_rfa(name: str = "wiki-rfa") -> GraphData:
    """Wikipedia Requests for Adminship (West et al. 2014, SNAP wiki-RfA): votes
    (support +1 / oppose -1; neutral dropped) from voter to candidate; label =
    election result of the candidate (RES). LEAKAGE NOTE (Master Prompt §3.1 C5.3):
    the label is the outcome of the votes that form the candidate's incoming
    edges; the sign-of-summed-votes baseline (signed_vote) is mandatory."""
    path = _fetch(name, "wiki-RfA.txt.gz", "wiki-rfa/wiki-RfA.txt.gz")
    names: dict[str, int] = {}
    src, dst, sign = [], [], []
    label_of: dict[int, int] = {}
    rec: dict[str, str] = {}

    def flush() -> None:
        if {"SRC", "TGT", "VOT", "RES"} <= rec.keys() and rec["SRC"] and rec["TGT"]:
            a = names.setdefault(rec["SRC"], len(names))
            b = names.setdefault(rec["TGT"], len(names))
            res = int(rec["RES"])
            if res in (-1, 1):
                label_of[b] = res
            v = int(rec["VOT"])
            if v != 0:
                src.append(a)
                dst.append(b)
                sign.append(float(v))

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                rec = {}
                continue
            if ":" in line[:4]:
                k, _, v = line.partition(":")
                rec[k.strip()] = v.strip()
        flush()
    return _signed_graph(name, None, np.array(src), np.array(dst), np.array(sign), label_of,
                         "SNAP wiki-RfA", {"leakage_note": "labels are election outcomes of incoming votes"})


def wiki_elec(name: str = "wiki-elec") -> GraphData:
    """Wikipedia adminship elections 2004-2008 (Leskovec et al., SNAP wikiElec.ElecBs3):
    E (promoted 1 / not 0), U (candidate), V (vote, voter id); votes +1/-1 kept,
    neutral dropped; label = E of the candidate's election (last one if repeated).
    Same LEAKAGE NOTE as wiki-rfa."""
    path = _fetch(name, "wikiElec.ElecBs3.txt.gz", "wiki-elec/wikiElec.ElecBs3.txt.gz")
    src, dst, sign = [], [], []
    label_of: dict[int, int] = {}
    result, cand = None, None
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            tag = parts[0].strip()
            if tag == "E":
                result = int(parts[1])
            elif tag == "U":
                cand = int(parts[1])
                label_of[cand] = int(result) if result is not None else -1
            elif tag == "V" and cand is not None:
                v, voter = int(parts[1]), int(parts[2])
                if v != 0:
                    src.append(voter)
                    dst.append(cand)
                    sign.append(float(v))
    label_of = {k: v for k, v in label_of.items() if v in (0, 1)}
    return _signed_graph(name, None, np.array(src), np.array(dst), np.array(sign), label_of,
                         "SNAP wikiElec.ElecBs3", {"leakage_note": "labels are election outcomes of incoming votes"})


LOADERS.update({
    "yelpchi": fraud_mat,
    "amazon-fraud": fraud_mat,
    "wiki-editor": wiki_editor,
    "wiki-rfa": wiki_rfa,
    "wiki-elec": wiki_elec,
})


# --------------------------------------------------- C5 Congress signed networks
# Neal (2020), "A sign of the times? Weak and strong polarization in the U.S.
# Congress, 1973-2016", Social Networks; figshare 10.6084/m9.figshare.8096429.
# One workbook, one tab per chamber and session ("H93", "S114", ...): a square signed
# adjacency matrix (+1 / -1 / 0) inferred from bill co-sponsorship with the SDSM
# backbone; the header row names each legislator with party and state. The signs
# are inferred from co-sponsorship only, so the party label is not used to build
# the graph (no label leakage by construction).
CONGRESS_SESSIONS = (93, 100, 107, 114)  # registered subset, one per ~14 years (PREREGISTRATION A6)
_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_sheets(path: Path) -> dict[str, list[list[str]]]:
    """Minimal XLSX reader (zip + XML; no dependency): {sheet name: rows of cell strings}."""
    import xml.etree.ElementTree as ET

    def col_index(ref: str) -> int:
        letters = re.match(r"[A-Z]+", ref).group(0)  # type: ignore[union-attr]
        n = 0
        for ch in letters:
            n = n * 26 + ord(ch) - 64
        return n - 1

    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _XLSX_NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_XLSX_NS['m']}}}t")))
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        out = {}
        for sh in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _XLSX_NS):  # type: ignore[union-attr]
            target = rels[sh.get(rel_ns)].lstrip("/")
            target = target if target.startswith("xl/") else "xl/" + target
            rows = []
            for row in ET.fromstring(z.read(target)).iter(f"{{{_XLSX_NS['m']}}}row"):
                cells: dict[int, str] = {}
                for c in row.findall("m:c", _XLSX_NS):
                    t, v = c.get("t"), c.find("m:v", _XLSX_NS)
                    if t == "s" and v is not None:
                        val = shared[int(v.text)]  # type: ignore[arg-type]
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(f"{{{_XLSX_NS['m']}}}t"))
                    else:
                        val = v.text if v is not None and v.text is not None else ""
                    cells[col_index(c.get("r"))] = val  # type: ignore[arg-type]
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
            out[sh.get("name")] = rows
    return out


def _party_of(label: str) -> str | None:
    """Party letter from a legislator label: 'Baucus, M. (MT-D)' (the figshare file), 'Smith (D-NY)', 'SMITH_D_NY'."""
    for pat in (r"\([A-Z]{2}-([A-Z])\)", r"\(([A-Z])\s*[-,]", r"\[([A-Z])\s*-", r"(?:^|[\s_,-])([DRI])(?:[\s_,-]+[A-Z]{2})(?:\W*$|[\s_)])",
                r"(?:^|\W)(Democrat|Republican|Independent)"):
        m = re.search(pat, label)
        if m:
            return m.group(1)[0]
    return None


def congress(name: str) -> GraphData:
    """congress-{house,senate}-<session>: party (D / R) labels; independents and
    unparseable labels are unlabelled (-1) and stay in the graph."""
    chamber, session = name.split("-")[1], int(name.split("-")[2])
    meta_path = _fetch(name, "8096429", "congress/figshare-8096429.json")
    files = json.loads(meta_path.read_text())["files"]
    xf = next(f for f in files if f["name"].endswith(".xlsx"))
    path = acquisition.fetch(xf["download_url"], "congress/" + xf["name"], dataset=name, cell_id=cell_of(name),
                             candidate_rank=1)
    sheets = _xlsx_sheets(path)
    tab = ("H" if chamber == "house" else "S") + str(session)
    key = next((s for s in sheets if s.replace(" ", "").upper() == tab), None)
    if key is None:
        raise ValueError(f"no sheet {tab}; sheets: {sorted(sheets)[:8]}…")
    rows = [r for r in sheets[key] if any(x != "" for x in r)]
    if not _is_number(rows[0][0]) and len(rows[0]) > 1 and _is_number(rows[0][1]):
        # observed layout of the figshare file: no header row, column A = "Name (ST-P)"
        names = [r[0] for r in rows]
        body = [r[1:] for r in rows]
    else:  # header-row layout, with or without a row-label column
        header = rows[0]
        has_row_labels = bool(len(rows) > 1 and rows[1] and not _is_number(rows[1][0]))
        names = header[1:] if has_row_labels or header[0] == "" else header
        body = [r[1:] if has_row_labels else r for r in rows[1:]]
    n = len(names)
    M = np.zeros((n, n))
    for i, r in enumerate(body[:n]):
        vals = [float(x) if _is_number(x) else 0.0 for x in r[:n]]
        M[i, : len(vals)] = vals
    parties = [_party_of(x) for x in names]
    parsed = sum(p is not None for p in parties)
    if parsed < 0.9 * n:
        with zipfile.ZipFile(path) as z:
            names_in_zip = z.namelist()[:12]
            first_sheet = next((x for x in z.namelist() if x.startswith("xl/worksheets/")), "")
            raw = z.read(first_sheet)[:700].decode("utf-8", "replace") if first_sheet else ""
        raise ValueError(f"party parsed for only {parsed}/{n} labels; first labels: {names[:5]}; "
                         f"raw rows: {len(sheets[key])}, first raw row: {sheets[key][0][:4] if sheets[key] else None}; "
                         f"zip: {names_in_zip}; {first_sheet}: {raw!r}")
    cls = {"D": 0, "R": 1}
    labels = np.array([cls.get(p, -1) for p in parties], dtype=np.int64)  # type: ignore[arg-type]
    iu, ju = np.nonzero(np.triu(M, 1))
    sign = M[iu, ju]
    sign = np.where(sign != 0, sign, M.T[iu, ju])
    edges, _, esign, report = canonicalize_edges(np.stack([iu, ju], 1), n, edge_sign=sign)
    splits = random_splits(labels, 10, (0.6, 0.2, 0.2), seed=0)
    report |= {"sheet": key, "header_examples": names[:3], "party_counts": {p: parties.count(p) for p in set(parties)},
               "asymmetric_cells": int(np.sum(M != M.T)), "source": "figshare 10.6084/m9.figshare.8096429"}
    g = GraphData(name, n, edges, 2, splits, labels, edge_sign=esign)
    return _finish(g, report)


def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


for _ch in ("house", "senate"):
    for _s in CONGRESS_SESSIONS:
        LOADERS[f"congress-{_ch}-{_s}"] = congress


# ------------------------------------- PR-20 extension "native-mixed" (REGISTRY "extension")
# Candidate real graphs whose node labels mix natively and close to randomly across
# edges (k = 2, edge homophily expected near 0.5), plus actor (k = 5, low homophily);
# their homophily is measured by dataset-stats.yml, not assumed. Registered with
# "extension": "native-mixed": e1.yml leaves them out of its default E1/E2 lists, so
# they run only when named in its `datasets` input. Sources are pinned to commits;
# every download is TO VERIFY on the first Actions run (dataset-stats.yml).
LINKX_COMMIT = "82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6"  # CUAI/Non-Homophily-Large-Scale
GEOM_GCN_COMMIT = "1124af17444d7fd09686504ec46caa3f39f4f632"  # graphdml-uiuc-jlu/geom-gcn
FB100_SCHOOLS = {
    "fb100-penn94": "Penn94",
    "fb100-amherst41": "Amherst41",
    "fb100-cornell5": "Cornell5",
    "fb100-johnshopkins55": "Johns Hopkins55",
    "fb100-reed98": "Reed98",
}
TWITCH_LANGS = {"twitch-de": "DE", "twitch-en": "ENGB", "twitch-es": "ES", "twitch-fr": "FR",
                "twitch-pt": "PTBR", "twitch-ru": "RU"}
# Dense float32 feature matrices above this size are not materialised (GraphData holds
# dense features only); the graph then has features=None and Track B is NOT-RUN on it.
MAX_DENSE_FEATURE_BYTES = 1 << 30


def _dense_features(X, report: dict) -> np.ndarray | None:
    """Dense float32 copy of ``X`` (dense or scipy sparse), or None above MAX_DENSE_FEATURE_BYTES."""
    n, d = X.shape
    need = int(n) * int(d) * 4
    if need > MAX_DENSE_FEATURE_BYTES:
        report["features_omitted"] = {"shape": [int(n), int(d)], "dense_float32_bytes": need,
                                      "limit_bytes": MAX_DENSE_FEATURE_BYTES,
                                      "reason": "dense matrix too large; Track B is NOT-RUN on this graph"}
        return None
    return (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32, copy=False)


def _label_binarize_columns(cols: np.ndarray) -> tuple[sp.csr_matrix, list[int]]:
    """Per-column encoding of LINKX's ``load_fb100_dataset`` (sklearn ``label_binarize``
    with ``classes=np.unique(col)``): >= 3 values -> one-hot; 2 values -> one indicator
    column of the larger value; 1 value -> one all-zero column. Value 0 ("missing" in
    FB100) is a category like any other."""
    blocks, widths = [], []
    n = cols.shape[0]
    for j in range(cols.shape[1]):
        vals, inv = np.unique(cols[:, j], return_inverse=True)
        if len(vals) >= 3:
            blocks.append(sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), inv)), shape=(n, len(vals))))
        else:
            blocks.append(sp.csr_matrix((inv == 1).astype(np.float32).reshape(-1, 1)))
        widths.append(blocks[-1].shape[1])
    return sp.hstack(blocks, format="csr"), widths


def fb100(name: str) -> GraphData:
    """Facebook100 school networks (Traud, Mucha & Porter 2012), the five schools shipped
    in the LINKX repository (Lim et al., NeurIPS 2021): Penn94, Amherst41, Cornell5,
    Johns Hopkins55, Reed98.

    Source: ``data/facebook100/<school>.mat`` of CUAI/Non-Homophily-Large-Scale at commit
    LINKX_COMMIT (not Git LFS); keys ``A`` (sparse friendship adjacency) and
    ``local_info`` (columns: student/faculty status, gender, major, second major/minor,
    dorm/house, year, high school; 0 = missing). License: none stated in the LINKX
    repository; FB100 was released by Facebook to Traud et al. for research (TO VERIFY).
    Label: gender = ``local_info[:, 1] - 1`` as in LINKX (codes 1/2 -> classes 0/1; the
    data do not say which code is which gender); 0 (missing) -> -1, kept in the graph,
    excluded from splits and statistics. Features (Track B): LINKX's encoding of every
    other ``local_info`` column (:func:`_label_binarize_columns`); PyG's LINKXDataset
    one-hot encodes binary columns with two columns instead (differs only for columns
    with <= 2 distinct values). Headline metric: accuracy, as LINKX reports for fb100
    (its main.py uses ROC-AUC only for yelp-chi, twitch-e, ogbn-proteins and genius).
    Splits: 10 stratified random 60/20/20 (seeds 0-9) over labelled nodes
    (PREREGISTRATION §4; LINKX's fixed 5 x 50/25/25 Penn94 splits are not used, see
    docs/SPEC_AMBIGUITIES.md).
    """
    from scipy.io import loadmat

    school = FB100_SCHOOLS[name]
    path = _fetch(name, "facebook100/" + school.replace(" ", "%20") + ".mat",
                  "linkx/facebook100/" + school.replace(" ", "") + ".mat")
    d = loadmat(path)
    A = sp.coo_matrix(d["A"])
    info = np.asarray(d["local_info"]).astype(np.int64)
    del d
    n = A.shape[0]
    if A.shape != (n, n) or info.shape[0] != n or info.shape[1] < 2:
        raise ValueError(f"{name}: unexpected shapes A {A.shape}, local_info {info.shape}")
    labels = info[:, 1] - 1  # gender label, -1 means unlabelled (LINKX)
    edges, _, _, report = canonicalize_edges(np.stack([A.row, A.col], axis=1), n)
    del A
    X, widths = _label_binarize_columns(np.delete(info, 1, axis=1))
    report |= {"source": f"CUAI/Non-Homophily-Large-Scale@{LINKX_COMMIT[:7]} data/facebook100/{school}.mat",
               "label": "local_info[:, 1] - 1 (gender; 0 = missing -> -1)",
               "gender_code_counts": {int(v): int(c) for v, c in zip(*np.unique(info[:, 1], return_counts=True), strict=True)},
               "unlabelled_nodes": int(np.sum(labels < 0)),
               "feature_columns": "local_info columns 0, 2..6 (all but gender), LINKX label_binarize encoding",
               "feature_widths": widths}
    feats = _dense_features(X, report)
    g = GraphData(name, n, edges, int(labels.max() + 1), random_splits(labels, 10, (0.6, 0.2, 0.2), 0), labels,
                  features=feats, meta={"source": "LINKX facebook100 (Traud et al. 2012)", "domain": "social",
                                        "extension": "native-mixed"})
    return _finish(g, report)


def genius(name: str = "genius") -> GraphData:
    """genius (Lim & Benson, ICWSM 2021), as distributed with LINKX (Lim et al. 2021).

    Source: ``data/genius.mat`` of CUAI/Non-Homophily-Large-Scale at LINKX_COMMIT (39 MB,
    not Git LFS); keys ``edge_index`` (2 x m), ``node_feat`` (n x 12), ``label`` (as read
    by LINKX's ``load_genius``). Nodes are users of genius.com, edges follow relations;
    label: binary user flag of the LINKX benchmark ("marked" accounts; exact meaning TO
    VERIFY against Lim & Benson 2021). License: none stated in the LINKX repository (TO
    VERIFY). Headline metric: ROC-AUC, as in LINKX (imbalanced binary labels).
    Memory: ~421k nodes and ~1M edges are read as an edge list (no dense n x n object);
    the int64 edge array is ~16 MB, so the repository-wide int64 convention is kept.
    Splits: 10 stratified random 60/20/20 (seeds 0-9); LINKX's fixed 5 x 50/25/25 splits
    are not used (docs/SPEC_AMBIGUITIES.md).
    """
    from scipy.io import loadmat

    d = loadmat(_fetch(name, "genius.mat", "linkx/genius.mat"))
    labels = np.asarray(d["label"]).ravel().astype(np.int64)
    n = len(labels)
    ei = np.asarray(d["edge_index"])
    if ei.ndim != 2 or 2 not in ei.shape:
        raise ValueError(f"genius: unexpected edge_index shape {ei.shape}")
    ei = ei if ei.shape[0] == 2 else ei.T
    feats = np.asarray(d["node_feat"], dtype=np.float32)
    del d
    edges, _, _, report = canonicalize_edges(ei.T, n)
    del ei
    report |= {"source": f"CUAI/Non-Homophily-Large-Scale@{LINKX_COMMIT[:7]} data/genius.mat",
               "label_counts": {int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True), strict=True)}}
    g = GraphData(name, n, edges, int(labels.max() + 1), random_splits(labels, 10, (0.6, 0.2, 0.2), 0), labels,
                  features=feats, meta={"source": "LINKX genius (Lim & Benson 2021)", "domain": "social",
                                        "extension": "native-mixed"})
    return _finish(g, report)


def deezer_europe(name: str = "deezer-europe") -> GraphData:
    """Deezer Europe (Rozemberczki & Sarkar, CIKM 2020), the LINKX copy.

    Source: ``data/deezer-europe.mat`` of CUAI/Non-Homophily-Large-Scale at LINKX_COMMIT;
    keys ``A`` (sparse mutual-follower adjacency), ``label`` (gender) and ``features``
    (sparse binary liked-artist indicators), as read by LINKX's ``load_deezer_dataset``.
    PyG's DeezerEurope reads a different file (graphmining.ai npz, 128 features) that is
    not reachable from the development sandbox; it is not used. License: none stated in
    the LINKX repository; the original release (benedekrozemberczki/FEATHER) carries an
    MIT LICENSE file (whether it covers the data: TO VERIFY). Headline metric: accuracy, as LINKX reports for deezer-europe (binary gender).
    Features are kept only if the dense float32 matrix fits MAX_DENSE_FEATURE_BYTES
    (otherwise features=None, recorded in ``meta['canonicalization']``).
    Splits: 10 stratified random 60/20/20 (seeds 0-9).
    """
    from scipy.io import loadmat

    d = loadmat(_fetch(name, "deezer-europe.mat", "linkx/deezer-europe.mat"))
    labels = np.asarray(d["label"]).ravel().astype(np.int64)
    n = len(labels)
    A = sp.coo_matrix(d["A"])
    if A.shape != (n, n):
        raise ValueError(f"deezer-europe: A has shape {A.shape}, {n} labels")
    X = d["features"]
    del d
    edges, _, _, report = canonicalize_edges(np.stack([A.row, A.col], axis=1), n)
    report |= {"source": f"CUAI/Non-Homophily-Large-Scale@{LINKX_COMMIT[:7]} data/deezer-europe.mat",
               "label_counts": {int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True), strict=True)}}
    feats = _dense_features(X, report)
    g = GraphData(name, n, edges, int(labels.max() + 1), random_splits(labels, 10, (0.6, 0.2, 0.2), 0), labels,
                  features=feats, meta={"source": "LINKX deezer-europe (Rozemberczki & Sarkar 2020)",
                                        "domain": "social", "extension": "native-mixed"})
    return _finish(g, report)


def twitch(name: str) -> GraphData:
    """Twitch user networks DE, EN(GB), ES, FR, PT(BR), RU (Rozemberczki, Allen & Sarkar,
    "Multi-scale Attributed Node Embedding", J. Complex Networks 2021; the "twitch-e"
    graphs of LINKX).

    Source: the raw MUSAE files ``data/twitch/<LANG>/musae_<LANG>_{target.csv, edges.csv,
    features.json}`` of CUAI/Non-Homophily-Large-Scale at LINKX_COMMIT, parsed as LINKX's
    ``load_twitch``: node id = ``new_id`` column, label = ``mature`` == "True" (the
    streamer uses explicit language), duplicate target rows (the FR file) keep the first
    row, features = binary indicators of the ids listed in features.json with all-zero
    columns dropped. PyG's Twitch reads graphmining.ai npz files instead (128 features,
    not reachable from the development sandbox; not used). License: none stated in the
    LINKX repository; the original release (benedekrozemberczki/MUSAE) carries a GPL-3.0
    LICENSE file (whether it covers the data: TO VERIFY). Headline metric: ROC-AUC,
    as LINKX reports for twitch-e. Splits: 10 stratified random 60/20/20 (seeds 0-9).
    """
    lang = TWITCH_LANGS[name]
    paths = {k: _fetch(name, f"twitch/{lang}/musae_{lang}_{k}", f"linkx/twitch/{lang}/musae_{lang}_{k}")
             for k in ("target.csv", "edges.csv", "features.json")}
    with open(paths["target.csv"], newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header = [h.strip().lower() for h in rows[0]]
    c_id = header.index("new_id") if "new_id" in header else 5  # LINKX: row[5]
    c_y = header.index("mature") if "mature" in header else 2  # LINKX: row[2] == "True"
    label_of: dict[int, int] = {}
    duplicates = conflicting = 0
    for r in rows[1:]:
        i, y = int(r[c_id]), int(r[c_y].strip() == "True")
        if i in label_of:
            duplicates += 1
            conflicting += int(label_of[i] != y)
            continue
        label_of[i] = y
    n = len(label_of)
    if sorted(label_of) != list(range(n)):
        raise ValueError(f"{name}: new_id values are not 0..{n - 1}")
    labels = np.array([label_of[i] for i in range(n)], dtype=np.int64)
    with open(paths["edges.csv"], encoding="utf-8") as fh:
        next(fh)
        raw = np.array(fh.read().replace(",", " ").split(), dtype=np.int64).reshape(-1, 2)
    edges, _, _, report = canonicalize_edges(raw, n)
    feat_lists = json.loads(paths["features.json"].read_text(encoding="utf-8"))
    rr = [int(k) for k, v in feat_lists.items() for _ in v if int(k) < n]
    cc = [int(f) for k, v in feat_lists.items() for f in v if int(k) < n]
    width = (max(cc) + 1) if cc else 0
    X = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n, width))
    X.sum_duplicates()
    X.data[:] = 1.0
    keep = np.flatnonzero(np.asarray(X.sum(axis=0)).ravel() > 0)
    report |= {"source": f"CUAI/Non-Homophily-Large-Scale@{LINKX_COMMIT[:7]} data/twitch/{lang}",
               "duplicate_target_rows": duplicates, "conflicting_duplicate_labels": conflicting,
               "feature_keys_out_of_range": sum(int(k) >= n for k in feat_lists),
               "feature_columns_kept": len(keep),
               "label_counts": {int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True), strict=True)}}
    feats = _dense_features(X[:, keep], report)
    g = GraphData(name, n, edges, 2, random_splits(labels, 10, (0.6, 0.2, 0.2), 0), labels, features=feats,
                  meta={"source": f"LINKX twitch-e {lang} (Rozemberczki et al. 2021)", "domain": "social",
                        "extension": "native-mixed"})
    return _finish(g, report)


def actor(name: str = "actor") -> GraphData:
    """actor ("film"): actor-only induced subgraph of the film-director-actor-writer
    network (Tang et al. 2009) as built by Geom-GCN (Pei et al., ICLR 2020).

    Source: graphdml-uiuc-jlu/geom-gcn at GEOM_GCN_COMMIT, ``new_data/film/
    out1_node_feature_label.txt`` (header line, then ``id<TAB>f1,f2,...<TAB>label``),
    ``new_data/film/out1_graph_edges.txt`` (header line, then ``u<TAB>v``) and the fixed
    splits ``splits/film_split_0.6_0.2_{0..9}.npz`` (``train_mask``/``val_mask``/
    ``test_mask``), the files PyG's ``Actor`` reads. Edges: co-occurrence on a Wikipedia
    page; features: 932 binary keyword indicators (width as in Geom-GCN's
    ``utils_data.py``); label: one of 5 classes derived from the words of the actor's
    Wikipedia page. License: none stated in the Geom-GCN repository (TO VERIFY). Headline
    metric: accuracy (Geom-GCN and every later actor benchmark). Splits: the 10 official
    Geom-GCN splits; despite the "0.6_0.2" name they are 48/32/20 (TO VERIFY on the
    first run from the mask sizes, recorded in ``meta['canonicalization']``).
    """
    nf = _fetch(name, "new_data/film/out1_node_feature_label.txt", "geom-gcn/film/out1_node_feature_label.txt")
    ef = _fetch(name, "new_data/film/out1_graph_edges.txt", "geom-gcn/film/out1_graph_edges.txt")
    split_files = [_fetch(name, f"splits/film_split_0.6_0.2_{i}.npz", f"geom-gcn/splits/film_split_0.6_0.2_{i}.npz")
                   for i in range(10)]
    ids, rows, cols, ys = [], [], [], []
    for line in nf.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        nid, feats_s, lab = line.rstrip().split("\t")
        ids.append(int(nid))
        ys.append(int(lab))
        f = [int(x) for x in feats_s.split(",") if x != ""]
        rows += [int(nid)] * len(f)
        cols += f
    n = len(ids)
    if sorted(ids) != list(range(n)):
        raise ValueError(f"actor: node ids are not 0..{n - 1}")
    labels = np.empty(n, dtype=np.int64)
    labels[np.asarray(ids)] = ys
    width = max(932, (max(cols) + 1) if cols else 0)
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, width))
    X.sum_duplicates()
    X.data[:] = 1.0
    with open(ef, encoding="utf-8") as fh:
        next(fh)
        raw = np.array(fh.read().split(), dtype=np.int64).reshape(-1, 2)
    edges, _, _, report = canonicalize_edges(raw, n)
    masks = {k: [] for k in ("train_mask", "val_mask", "test_mask")}
    for p in split_files:
        with np.load(p) as z:
            for k, acc in masks.items():
                m = np.asarray(z[k]).astype(bool).ravel()
                if len(m) != n:
                    raise ValueError(f"actor: {p.name} {k} has length {len(m)}, expected {n}")
                acc.append(m)
    splits = _mask_splits(np.stack(masks["train_mask"]), np.stack(masks["val_mask"]), np.stack(masks["test_mask"]),
                          "official Geom-GCN film splits (film_split_0.6_0.2_{0..9})")
    report |= {"source": f"graphdml-uiuc-jlu/geom-gcn@{GEOM_GCN_COMMIT[:7]} new_data/film",
               "split_sizes": [[len(s.train_idx), len(s.val_idx), len(s.test_idx)] for s in splits],
               "label_counts": np.bincount(labels).tolist()}
    feats = _dense_features(X, report)
    g = GraphData(name, n, edges, int(labels.max() + 1), splits, labels, features=feats,
                  meta={"source": "Geom-GCN film (Pei et al. 2020)", "domain": "web/wiki (co-occurrence)",
                        "extension": "native-mixed"})
    return _finish(g, report)


LOADERS.update({
    **dict.fromkeys(FB100_SCHOOLS, fb100),
    **dict.fromkeys(TWITCH_LANGS, twitch),
    "genius": genius,
    "deezer-europe": deezer_europe,
    "actor": actor,
})


# ------------------------------------------- A18 extension (REGISTRY "extension": "a18")
# Real graphs this project had never used, registered as the confirmatory family of
# PREREGISTRATION A18: GraphLand (hm-categories, artnet-exp, city-reviews, web-fraud,
# web-topics), OGB ogbn-arxiv / ogbn-products, LINKX arxiv-year, twitch-gamers,
# snap-patents, pokec. They carry "extension": "a18" and "opt_in": true, so no default
# dataset list (dataset_stats.py, run_e1.py, edge_classifier_report.py, e1.yml) picks
# them up: they are processed only when named. Every download is exercised first by
# dataset-stats.yml in Actions; the formats below come from the authors' loader code.
GRAPHLAND_RECORD = "16895532"  # Zenodo record of GraphLand v1 (Apache-2.0)
GRAPHLAND = {  # name -> (task, headline metric of the GraphLand code: accuracy / AP)
    "hm-categories": ("multiclass_classification", "accuracy"),
    "artnet-exp": ("binary_classification", "average_precision"),
    "city-reviews": ("binary_classification", "average_precision"),
    "web-fraud": ("binary_classification", "average_precision"),
    "web-topics": ("multiclass_classification", "accuracy"),
}
GRAPHLAND_SPLITS = ("RH", "RL", "TH")  # provided split files; TH is absent for city-reviews
_GL_SPLIT_SOURCE = {
    "RH": "GraphLand RH (random high, 50/25/25)",
    "RL": "GraphLand RL (random low, 10/10/80, stratified)",
    "TH": "GraphLand TH (temporal high, 50/25/25)",
}
_SKLEARN_BOUNDS_THRESHOLD = 1e-7  # sklearn.preprocessing._data.BOUNDS_THRESHOLD


def _parse_simple_yaml(text: str) -> dict:
    """GraphLand ``info.yaml`` without a YAML dependency (PyYAML is used when installed).

    Supports what PyYAML's ``safe_dump`` writes for a flat mapping: ``key: scalar``,
    ``key: [a, b]`` and ``key:`` followed by ``- item`` lines; quoted scalars are
    unquoted. Nested mappings are not supported (GraphLand's info.yaml has none:
    every key the loaders read is a scalar or a list of strings)."""
    try:
        import yaml  # type: ignore[import-untyped]

        return dict(yaml.safe_load(text))
    except ImportError:
        pass

    def scalar(v: str):
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == "'":
            return v[1:-1].replace("''", "'")
        if len(v) >= 2 and v[0] == v[-1] == '"':
            return json.loads(v)
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("null", "~", ""):
            return None
        for conv in (int, float):
            try:
                return conv(v)
            except ValueError:
                pass
        return v

    out: dict = {}
    key = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped == "-" or stripped.startswith("- "):
            if key is None or not isinstance(out.get(key), list):
                raise ValueError(f"info.yaml: list item outside a list: {raw!r}")
            out[key].append(scalar(stripped[1:]))
        elif not raw[0].isspace() and ":" in raw:
            k, _, v = raw.partition(":")
            key, v = k.strip(), v.strip()
            if v == "":
                out[key] = []
            elif v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                out[key] = [scalar(x) for x in next(csv.reader([inner], skipinitialspace=True, quotechar="'"))] if inner else []
            else:
                out[key] = scalar(v)
        elif key is not None and isinstance(out.get(key), str):  # folded continuation of a long scalar
            out[key] = out[key] + " " + stripped
        else:
            raise ValueError(f"info.yaml: unsupported line {raw!r}")
    return out


def _numeric_lines(lines: list[str], dtype=np.float64) -> np.ndarray:
    """Parse comma-separated numeric lines (pandas ``to_csv`` output): empty fields are
    NaN, ``True``/``False`` are 1/0. Returns a 2-D array."""
    text = "".join(lines)
    if "\r" in text:
        text = text.replace("\r", "")
    text = text.replace(",,", ",nan,").replace(",,", ",nan,").replace(",\n", ",nan\n").replace("\n,", "\nnan,")
    if text.endswith(","):
        text += "nan"
    if text.startswith(","):
        text = "nan" + text
    if "True" in text or "False" in text:
        text = text.replace("True", "1").replace("False", "0")
    return np.loadtxt(io.StringIO(text), delimiter=",", dtype=dtype, ndmin=2)


def _csv_header(fh) -> list[str]:
    return next(csv.reader([fh.readline()]))


def _numeric_csv(fh, *, hint: int = 1 << 26, on_block: Callable[[np.ndarray], None] | None = None):
    """Header and float64 body of a numeric CSV; with ``on_block`` the body is streamed
    block by block to the callback instead of being returned."""
    header = _csv_header(fh)
    blocks = []
    while lines := fh.readlines(hint):
        lines = [x for x in lines if x.strip()]
        if not lines:
            continue
        block = _numeric_lines(lines)
        if on_block is None:
            blocks.append(block)
        else:
            on_block(block)
    if on_block is not None:
        return header, None
    body = np.concatenate(blocks) if blocks else np.empty((0, len(header)))
    return header, body


def _most_frequent(col: np.ndarray) -> float:
    """sklearn SimpleImputer(strategy='most_frequent'): the mode of the non-NaN values,
    the smallest one on ties."""
    vals, counts = np.unique(col[~np.isnan(col)], return_counts=True)
    return float(vals[np.argmax(counts)])


def _quantile_normal(col: np.ndarray, fit_col: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    """numpy port of sklearn ``QuantileTransformer(output_distribution='normal',
    n_quantiles=1000, subsample=None)``: fit on ``fit_col`` (NaN ignored), transform
    ``col`` (both float32, as PyG passes them). Parity with sklearn is checked by
    tests/unit/test_data.py when scikit-learn is installed."""
    from scipy.special import ndtri

    nq = min(n_quantiles, fit_col.shape[0])
    refs = np.linspace(0, 1, nq, endpoint=True)
    q = np.nanpercentile(fit_col, refs * 100)
    q = np.maximum.accumulate(q)
    x = np.array(col, copy=True)  # sklearn keeps the input dtype (float32) for the uniform scores
    with np.errstate(invalid="ignore"):
        lower = x - _SKLEARN_BOUNDS_THRESHOLD < q[0]
        upper = x + _SKLEARN_BOUNDS_THRESHOLD > q[-1]
    fin = ~np.isnan(x)
    xf = x[fin]
    x[fin] = 0.5 * (np.interp(xf, q, refs) - np.interp(-xf, -q[::-1], -refs[::-1]))
    x[upper] = 1
    x[lower] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        x = ndtri(x.astype(np.float64))  # scipy.stats.norm.ppf (sklearn) works in float64
        clip_min = ndtri(_SKLEARN_BOUNDS_THRESHOLD - np.spacing(1))
        clip_max = ndtri(1 - (_SKLEARN_BOUNDS_THRESHOLD - np.spacing(1)))
        return np.clip(x, clip_min, clip_max)


def _zip_member(zf: zipfile.ZipFile, fname: str, required: bool = True) -> str | None:
    hits = sorted((m for m in zf.namelist() if m.rsplit("/", 1)[-1] == fname and "__MACOSX" not in m), key=len)
    if not hits:
        if required:
            raise FileNotFoundError(f"{fname} not in {zf.filename}: {zf.namelist()[:20]}")
        return None
    return hits[0]


def _text(zf: zipfile.ZipFile, member: str):
    return io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="")


def _check_index(idx: np.ndarray, n: int, what: str) -> None:
    if len(idx) != n or not np.array_equal(idx, np.arange(n)):
        raise ValueError(f"{what}: index column is not 0..{n - 1}")


def _int_labels(values: np.ndarray, report: dict) -> tuple[np.ndarray, int]:
    """Class labels from float targets: NaN -> -1 (unlabelled); classes are kept as
    given when they are 0..k-1, else remapped to 0..k-1 (mapping recorded)."""
    lab = ~np.isnan(values)
    vals = values[lab]
    if not np.all(vals == np.round(vals)):
        raise ValueError("classification targets are not integers")
    classes = np.unique(vals).astype(np.int64)
    labels = np.full(len(values), -1, dtype=np.int64)
    if np.array_equal(classes, np.arange(len(classes))):
        labels[lab] = vals.astype(np.int64)
    else:
        labels[lab] = np.searchsorted(classes, vals.astype(np.int64))
        report["class_values_remapped"] = classes.tolist()
    report["unlabelled_nodes"] = int((~lab).sum())
    report["label_counts"] = np.bincount(labels[lab], minlength=len(classes)).tolist()
    return labels, len(classes)


def _graphland_features(zf: zipfile.ZipFile, info: dict, n: int, report: dict) -> np.ndarray | None:
    """GraphLand node features as PyG's ``GraphLandDataset`` builds them by default:
    [numerical (non-fraction) columns: most-frequent imputation, then
    QuantileTransformer(normal) fitted on all nodes (label-free, transductive);
    fraction columns: most-frequent imputation, no transform; categorical columns:
    one-hot with ``drop='if_binary'`` (NaN, if any, is one more category)]. Returns
    None above MAX_DENSE_FEATURE_BYTES (shape and reason in ``report``)."""
    member = _zip_member(zf, "features.csv")
    frac = list(info.get("fraction_features_names") or [])
    num = [c for c in (info.get("numerical_features_names") or []) if c not in set(frac)]
    cat = list(info.get("categorical_features_names") or [])
    with _text(zf, member) as fh:
        header = _csv_header(fh)
    cols = {c: i for i, c in enumerate(header)}
    missing = [c for c in num + frac + cat if c not in cols]
    if missing:
        raise ValueError(f"features.csv lacks info.yaml columns {missing[:10]} (header: {header[:10]})")
    pos_num = np.array([cols[c] for c in num], dtype=np.int64)
    pos_frac = np.array([cols[c] for c in frac], dtype=np.int64)
    pos_cat = np.array([cols[c] for c in cat], dtype=np.int64)
    nan_count = np.zeros(len(header), dtype=np.int64)
    cat_values: list[np.ndarray] = [np.empty(0) for _ in cat]
    keep = int(n) * (len(num) + len(frac) + len(cat)) * 4 <= MAX_DENSE_FEATURE_BYTES
    kept: list[np.ndarray] = []
    rows = [0]

    def on_block(b: np.ndarray) -> None:
        if b.shape[1] != len(header):
            raise ValueError(f"features.csv: {b.shape[1]} columns, header has {len(header)}")
        nan_count[:] += np.isnan(b).sum(axis=0)
        for j, p in enumerate(pos_cat):
            v = b[:, p]
            cat_values[j] = np.union1d(cat_values[j], np.unique(np.where(np.isnan(v), -(2.0**31), v)))
        rows[0] += len(b)
        if keep:
            kept.append(b.astype(np.float32))

    with _text(zf, member) as fh:
        _numeric_csv(fh, on_block=on_block)
    if rows[0] != n:
        raise ValueError(f"features.csv has {rows[0]} rows, targets.csv {n}")
    empty = [header[p] for p in np.concatenate([pos_num, pos_frac]) if nan_count[p] == n]
    widths = [1 if len(v) <= 2 else len(v) for v in cat_values]
    d = len(num) + len(frac) - len(empty) + sum(widths)
    report["features"] = {
        "numerical": len(num), "fraction": len(frac), "categorical": len(cat), "categorical_onehot_width": int(sum(widths)),
        "dimension": int(d), "all_nan_columns_dropped": empty,
        "nan_counts": {header[p]: int(nan_count[p]) for p in range(len(header)) if nan_count[p]},
        "preprocessing": "PyG GraphLandDataset defaults: numerical most_frequent imputation + QuantileTransformer"
                         "(normal, 1000 quantiles, fitted on all nodes); fraction most_frequent imputation; "
                         "categorical one-hot drop='if_binary'; order [numerical, fraction, categorical]",
    }
    need = int(n) * int(d) * 4
    if not keep or need > MAX_DENSE_FEATURE_BYTES:
        report["features_omitted"] = {"shape": [int(n), int(d)], "dense_float32_bytes": need,
                                      "limit_bytes": MAX_DENSE_FEATURE_BYTES,
                                      "reason": "dense matrix too large; Track B is NOT-RUN on this graph"}
        return None
    B = np.concatenate(kept)
    kept.clear()
    X = np.empty((n, d), dtype=np.float32)
    j = 0
    for p in pos_num:
        if nan_count[p] == n:
            continue
        col = B[:, p]
        fitted = col.copy()
        if nan_count[p]:
            col = np.where(np.isnan(col), np.float32(_most_frequent(col)), col)
        X[:, j] = _quantile_normal(col, fitted)
        j += 1
    for p in pos_frac:
        if nan_count[p] == n:
            continue
        col = B[:, p]
        X[:, j] = np.where(np.isnan(col), np.float32(_most_frequent(col)), col) if nan_count[p] else col
        j += 1
    for p, vals in zip(pos_cat, cat_values, strict=True):
        col = np.where(np.isnan(B[:, p]), np.float32(-(2.0**31)), B[:, p]).astype(np.float64)
        if len(vals) == 2:
            X[:, j] = col == vals[1]
            j += 1
        else:
            X[:, j:j + len(vals)] = col[:, None] == vals[None, :]
            j += len(vals)
    assert j == d
    return X


def graphland(name: str) -> GraphData:
    """GraphLand datasets (Bazhenov, Platonov & Prokhorenkova, NeurIPS 2025 D&B;
    Zenodo record 16895532, Apache-2.0): hm-categories, artnet-exp, city-reviews,
    web-fraud, web-topics.

    Source: ``<name>.zip`` of the Zenodo record (md5 from the record page, checked on
    download). Layout as read by PyG's ``GraphLandDataset`` and the authors'
    ``dataset.py`` (yandex-research/graphland@7246fe3): ``info.yaml`` (feature-name
    lists, ``target_name``, ``task``), ``features.csv`` and ``targets.csv`` (first
    column = node index), ``edgelist.csv`` (header; the first two columns are the
    endpoints), ``split_masks_{RL,RH,TH}.csv`` (columns train / val / test). Edges are
    symmetrised, self-loops and duplicates removed (web-fraud / web-topics are directed:
    §2.6). Targets that are NaN are unlabelled (-1) and excluded from every split, as in
    PyG (``mask & labeled_mask``). Features: :func:`_graphland_features`.
    Splits, in this order: the provided RH, RL and (where it exists) TH splits by name,
    then 10 stratified random 60/20/20 splits ``rand0``-``rand9`` (seeds 0-9, labelled
    nodes). Headline metric: the GraphLand code's (accuracy for multiclass, average
    precision for binary).
    """
    task, _metric = GRAPHLAND[name]
    fname = f"{name}.zip"
    path = _fetch(name, fname, f"graphland/{fname}", expected_md5=dataset_entry(name).get("md5", {}).get(fname))
    report: dict = {"source": f"Zenodo {GRAPHLAND_RECORD} {fname}"}
    with zipfile.ZipFile(path) as zf:
        info = _parse_simple_yaml(zf.read(_zip_member(zf, "info.yaml")).decode("utf-8"))
        if info.get("task") not in (None, task):
            raise ValueError(f"{name}: info.yaml task {info.get('task')!r}, expected {task!r}")
        with _text(zf, _zip_member(zf, "targets.csv")) as fh:
            t_header, T = _numeric_csv(fh)
        n = len(T)
        _check_index(T[:, 0], n, "targets.csv")
        tname = info.get("target_name")
        tcol = t_header.index(tname) if tname in t_header else (1 if len(t_header) == 2 else None)
        if tcol is None:
            raise ValueError(f"{name}: target column {tname!r} not in {t_header}")
        labels, k = _int_labels(T[:, tcol], report)
        del T
        with _text(zf, _zip_member(zf, "edgelist.csv")) as fh:
            e_header = _csv_header(fh)
            try:
                raw = np.loadtxt(fh, delimiter=",", usecols=(0, 1), dtype=np.int64, ndmin=2)
            except ValueError:
                raise ValueError(f"{name}: non-integer endpoints in edgelist.csv (header {e_header})") from None
        edges, _, _, rep = canonicalize_edges(raw, n)
        report |= rep | {"edgelist_header": e_header}
        del raw
        splits: list[Split] = []
        sizes = {}
        for sid in GRAPHLAND_SPLITS:
            member = _zip_member(zf, f"split_masks_{sid}.csv", required=False)
            if member is None:
                continue
            with _text(zf, member) as fh:
                m_header, M = _numeric_csv(fh)
            _check_index(M[:, 0], n, member)
            idx = {c: m_header.index(c) for c in ("train", "val", "test")}
            lab = labels >= 0
            tr, va, te = (np.flatnonzero((M[:, idx[c]] == 1) & lab) for c in ("train", "val", "test"))
            sizes[sid] = {"train": len(tr), "val": len(va), "test": len(te),
                          "unlabelled_in_masks": int(((M[:, 1:] == 1).any(axis=1) & ~lab).sum())}
            splits.append(Split(sid, tr, va, te, f"{_GL_SPLIT_SOURCE[sid]}, official, Zenodo {GRAPHLAND_RECORD}"))
        report["provided_splits"] = sizes
        feats = _graphland_features(zf, info, n, report)
    report["info_yaml"] = {k_: v for k_, v in info.items() if not k_.endswith("_names")}
    splits += random_splits(labels, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, k, splits, labels, features=feats,
                  meta={"source": f"GraphLand (Bazhenov et al. 2025), Zenodo {GRAPHLAND_RECORD}", "extension": "a18",
                        "task": task, "split_ids": [s.split_id for s in splits],
                        "symmetrised_from_directed": bool(info.get("graph_is_directed", name.startswith("web-")))})
    return _finish(g, report)


# --------------------------------------------------------------------------- OGB
def _loadtxt_gz(zf: zipfile.ZipFile, member: str, dtype) -> np.ndarray:
    with zf.open(member) as raw, gzip.open(raw, "rt") as fh:
        return np.loadtxt(fh, delimiter=",", dtype=dtype, ndmin=2)


def _ogb_zip(name: str, zipname: str) -> tuple[zipfile.ZipFile, Callable[[str], str]]:
    path = _fetch(name, zipname, f"ogb/{zipname}")
    zf = zipfile.ZipFile(path)

    def member(suffix: str) -> str:
        hits = sorted((m for m in zf.namelist() if m.endswith(suffix)), key=len)
        if not hits:
            raise FileNotFoundError(f"{suffix} not in {zipname}")
        return hits[0]

    return zf, member


def _ogb_graph(zf: zipfile.ZipFile, member: Callable[[str], str], report: dict) -> tuple[int, np.ndarray, np.ndarray]:
    raw = _loadtxt_gz(zf, member("raw/edge.csv.gz"), np.int64)
    feats = _loadtxt_gz(zf, member("raw/node-feat.csv.gz"), np.float32)
    n = int(_loadtxt_gz(zf, member("raw/num-node-list.csv.gz"), np.int64).ravel()[0])
    if feats.shape[0] != n:
        raise ValueError(f"node-feat has {feats.shape[0]} rows, num-node-list says {n}")
    edges, _, _, rep = canonicalize_edges(raw, n)
    report |= rep | {"raw_edge_rows": len(raw)}
    return n, edges, feats


def _ogb_split(zf: zipfile.ZipFile, member: Callable[[str], str], split_dir: str, source: str) -> Split:
    idx = [_loadtxt_gz(zf, member(f"split/{split_dir}/{s}.csv.gz"), np.int64).ravel() for s in ("train", "valid", "test")]
    return Split(split_dir, *(np.sort(i) for i in idx), source)


def ogbn_multiclass(name: str) -> GraphData:
    """ogbn-arxiv / ogbn-products (Hu et al., NeurIPS 2020) from the raw OGB archives
    (no ``ogb`` dependency; layout of ogb/nodeproppred/master.csv: ``<d>/raw/{edge,
    node-feat, node-label, num-node-list}.csv.gz``, ``<d>/split/<split>/{train, valid,
    test}.csv.gz``). arxiv: directed citations (``add_inverse_edge`` False), symmetrised
    here (§2.6, as the Master Prompt's registry note says); products: undirected
    co-purchases stored once per pair. Splits: the official one first (split_id
    ``time`` / ``sales_ranking``), then 10 stratified random 60/20/20 ``rand0``-``rand9``.
    Headline metric: accuracy (OGB). Licence: ODC-BY (arxiv: MAG) / Amazon terms
    (products) per the OGB documentation, TO VERIFY."""
    zipname, split_dir = {"ogbn-arxiv": ("arxiv.zip", "time"), "ogbn-products": ("products.zip", "sales_ranking")}[name]
    zf, member = _ogb_zip(name, zipname)
    report: dict = {"source": f"OGB {zipname}"}
    with zf:
        n, edges, feats = _ogb_graph(zf, member, report)
        labels = _loadtxt_gz(zf, member("raw/node-label.csv.gz"), np.float64).ravel()
        official = _ogb_split(zf, member, split_dir, f"official OGB {split_dir} split")
    labels, k = _int_labels(labels, report)
    report["official_split_sizes"] = [len(official.train_idx), len(official.val_idx), len(official.test_idx)]
    splits = [official] + random_splits(labels, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, k, splits, labels, features=_dense_features(feats, report),
                  meta={"source": f"OGB {name}", "extension": "a18", "split_ids": [s.split_id for s in splits],
                        "symmetrised_from_directed": name == "ogbn-arxiv"})
    return _finish(g, report)


# ------------------------------------------------------------------------- LINKX
def even_quantile_labels(vals: np.ndarray, nclasses: int) -> np.ndarray:
    """Exact port of LINKX's ``data_utils.even_quantile_labels`` (CUAI@82f8f05): class c
    holds the values in [q_c, q_{c+1}) of the c/nclasses quantiles (np.nanquantile,
    linear); the last class is closed above; NaN stays -1."""
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    lower = -np.inf
    for c in range(nclasses - 1):
        upper = np.nanquantile(vals, (c + 1) / nclasses)
        inds = (vals >= lower) * (vals < upper)
        label[inds] = c
        lower = upper
    label[vals >= lower] = nclasses - 1
    return label


class _SplitUnpickler(pickle.Unpickler):
    """Unpickler for LINKX ``*-splits.npy`` object arrays: numpy (and torch, for tensors)
    only; any other global is refused."""

    def find_class(self, module: str, name: str):
        root = module.split(".")[0]
        if root in ("numpy", "torch") or (module == "collections" and name == "OrderedDict") or (
                module == "builtins" and name in ("dict", "list", "tuple", "int", "float", "bool", "slice", "range")) or (
                module == "_codecs" and name == "encode"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name} in a splits file")


def _linkx_fixed_splits(path: Path, labels: np.ndarray, tag: str) -> tuple[list[Split], dict]:
    """The fixed LINKX splits (a list of {'train', 'valid', 'test'} index arrays, saved
    with np.save(allow_pickle=True)); indices of unlabelled nodes are dropped (LINKX's
    ``rand_train_test_idx`` never draws them)."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        _shape, _fortran, dtype = reader(f)
        if dtype.hasobject:
            obj = _SplitUnpickler(f, encoding="latin1").load()
        else:
            obj = None
    if obj is None:
        raise ValueError(f"{path.name}: expected an object array of split dicts, got dtype {dtype}")
    n = len(labels)
    splits, rep = [], {"dropped_unlabelled": 0}
    for i, d in enumerate(np.asarray(obj, dtype=object).ravel()):
        parts = []
        for key in ("train", "valid" if "valid" in d else "val", "test"):
            a = np.asarray(d[key])
            a = np.flatnonzero(a) if a.dtype == bool else a.astype(np.int64).ravel()
            if a.size and (a.min() < 0 or a.max() >= n):
                raise ValueError(f"{path.name}: split {i} {key} index out of range")
            keep = labels[a] >= 0
            rep["dropped_unlabelled"] += int((~keep).sum())
            parts.append(np.sort(a[keep]))
        splits.append(Split(f"linkx{i}", *parts, f"LINKX fixed split {i} ({tag}), 50/25/25"))
    rep["sizes"] = [[len(s.train_idx), len(s.val_idx), len(s.test_idx)] for s in splits]
    return splits, rep


def arxiv_year(name: str = "arxiv-year") -> GraphData:
    """arxiv-year (Lim et al., NeurIPS 2021): ogbn-arxiv's graph and features (same raw
    archive and symmetrisation as :func:`ogbn_multiclass`), label = publication year in
    5 classes by LINKX's :func:`even_quantile_labels` of ``raw/node_year.csv.gz``.
    Splits: LINKX's 5 fixed 50/25/25 splits (``data/splits/arxiv-year-splits.npy`` of
    CUAI/Non-Homophily-Large-Scale at LINKX_COMMIT, ``linkx0``-``linkx4``), then 10
    stratified random 60/20/20 ``rand0``-``rand9``. Headline metric: accuracy (LINKX)."""
    zf, member = _ogb_zip(name, "arxiv.zip")
    report: dict = {"source": "OGB arxiv.zip (node_year) + LINKX even_quantile_labels"}
    with zf:
        n, edges, feats = _ogb_graph(zf, member, report)
        years = _loadtxt_gz(zf, member("raw/node_year.csv.gz"), np.int64).ravel()
    labels = even_quantile_labels(years, 5)
    report["year_range_per_class"] = [[int(years[labels == c].min()), int(years[labels == c].max())] for c in range(5)]
    report["label_counts"] = np.bincount(labels[labels >= 0], minlength=5).tolist()
    fixed, rep = _linkx_fixed_splits(_fetch_aux(name, "splits", "arxiv-year-splits.npy", "linkx/splits/arxiv-year-splits.npy"),
                                     labels, f"CUAI@{LINKX_COMMIT[:7]} data/splits/arxiv-year-splits.npy")
    report["linkx_splits"] = rep
    splits = fixed + random_splits(labels, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, 5, splits, labels, features=_dense_features(feats, report),
                  meta={"source": "LINKX arxiv-year (ogbn-arxiv graph, year labels)", "extension": "a18",
                        "split_ids": [s.split_id for s in splits], "symmetrised_from_directed": True})
    return _finish(g, report)


def twitch_gamers(name: str = "twitch-gamers") -> GraphData:
    """Twitch Gamers (Rozemberczki & Sarkar 2021, arXiv:2101.03091), the LINKX
    "twitch-gamers" task.

    Source: SNAP ``twitch_gamers.zip`` (``*features.csv`` and ``*edges.csv`` inside,
    TO VERIFY on the first run), or LINKX's Google Drive copies of the two CSVs
    (``aux_routes``). Parsed as LINKX's ``load_twitch_gamer_dataset(task='mature')``:
    drop ``numeric_id``; ``created_at`` / ``updated_at`` as the integer YYYYMMDD;
    ``language`` as an integer code by order of first appearance; label = ``mature``;
    features = the remaining 7 columns in file order, standardised over all nodes
    (mean, unbiased std; label-free). Node i = the row with ``numeric_id`` i (LINKX
    uses the row order; whether they coincide is recorded). Splits: LINKX's 5 fixed
    50/25/25 splits (``data/splits/twitch-gamer-splits.npy`` at LINKX_COMMIT), then 10
    stratified random 60/20/20. Headline metric: accuracy (LINKX)."""
    try:
        path = _fetch(name, "twitch_gamers.zip", "snap/twitch_gamers.zip")
        with zipfile.ZipFile(path) as zf:
            fm = sorted((m for m in zf.namelist() if m.endswith("features.csv") and "__MACOSX" not in m), key=len)
            em = sorted((m for m in zf.namelist() if m.endswith("edges.csv") and "__MACOSX" not in m), key=len)
            if not fm or not em:
                raise FileNotFoundError(f"twitch_gamers.zip members: {zf.namelist()}")
            feat_text = zf.read(fm[0]).decode("utf-8")
            with _text(zf, em[0]) as fh:
                fh.readline()
                raw = np.loadtxt(fh, delimiter=",", dtype=np.int64, ndmin=2)
        source = f"SNAP twitch_gamers.zip ({fm[0]}, {em[0]})"
    except acquisition.AcquisitionError as exc:
        fpath = _fetch_aux(name, "linkx_feat", "twitch-gamer_feat.csv", "linkx/twitch-gamer_feat.csv")
        epath = _fetch_aux(name, "linkx_edges", "twitch-gamer_edges.csv", "linkx/twitch-gamer_edges.csv")
        feat_text = fpath.read_text(encoding="utf-8")
        with open(epath, encoding="utf-8") as fh:
            fh.readline()
            raw = np.loadtxt(fh, delimiter=",", dtype=np.int64, ndmin=2)
        source = f"LINKX Google Drive twitch-gamer_{{feat,edges}}.csv (SNAP failed: {str(exc)[:200]})"
    rows = [r for r in csv.reader(io.StringIO(feat_text)) if r]
    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    n = len(body)
    col = {h: i for i, h in enumerate(header)}
    for need in ("numeric_id", "mature", "language", "created_at", "updated_at"):
        if need not in col:
            raise ValueError(f"{name}: column {need!r} missing; header {header}")
    ids = np.array([int(r[col["numeric_id"]]) for r in body], dtype=np.int64)
    if sorted(ids.tolist()) != list(range(n)):
        raise ValueError(f"{name}: numeric_id is not a permutation of 0..{n - 1}")
    lang_code: dict[str, int] = {}
    feat_cols = [h for h in header if h not in ("numeric_id", "mature")]
    F = np.empty((n, len(feat_cols)), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    for r, nid in zip(body, ids, strict=True):
        y[nid] = int(float(r[col["mature"]]))
        for j, h in enumerate(feat_cols):
            v = r[col[h]].strip()
            if h in ("created_at", "updated_at"):
                F[nid, j] = int(v.replace("-", ""))
            elif h == "language":
                F[nid, j] = lang_code.setdefault(v, len(lang_code))
            else:
                F[nid, j] = float(v)
    F = (F - F.mean(axis=0)) / F.std(axis=0, ddof=1)
    edges, _, _, report = canonicalize_edges(raw, n)
    del raw
    report |= {"source": source, "feature_columns": feat_cols, "languages": len(lang_code),
               "rows_sorted_by_numeric_id": bool(np.array_equal(ids, np.arange(n))),
               "label_counts": np.bincount(y, minlength=2).tolist(),
               "feature_preprocessing": "LINKX load_twitch_gamer + standardisation (mean, unbiased std) over all nodes"}
    fixed, rep = _linkx_fixed_splits(_fetch_aux(name, "splits", "twitch-gamer-splits.npy", "linkx/splits/twitch-gamer-splits.npy"),
                                     y, f"CUAI@{LINKX_COMMIT[:7]} data/splits/twitch-gamer-splits.npy")
    report["linkx_splits"] = rep
    splits = fixed + random_splits(y, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, 2, splits, y, features=F.astype(np.float32),
                  meta={"source": "Twitch Gamers (Rozemberczki & Sarkar 2021), LINKX task 'mature'",
                        "extension": "a18", "split_ids": [s.split_id for s in splits]})
    return _finish(g, report)


def _linkx_mat(name: str, fname: str) -> dict:
    from scipy.io import loadmat

    return loadmat(_fetch(name, fname, f"linkx/{fname}"))


def snap_patents(name: str = "snap-patents") -> GraphData:
    """snap-patents (Lim et al. 2021): utility patents 1963-1999 with citation edges
    (subset of SNAP cit-Patents / NBER), label = grant year in 5 classes by
    :func:`even_quantile_labels` of ``years``; LINKX's ``snap_patents.mat`` (keys
    ``edge_index``, ``node_feat`` sparse, ``num_nodes``, ``years``) from the authors'
    Google Drive (TO VERIFY). Directed citations are symmetrised (§2.6). Features are
    kept only within MAX_DENSE_FEATURE_BYTES (expected: 269 columns x 2.9M nodes ~ 3.1 GB
    -> omitted, Track B NOT-RUN). Splits: LINKX's 5 fixed splits when the registry lists
    an ``aux_routes['splits']`` source, then 10 stratified random 60/20/20.
    Headline metric: accuracy (LINKX)."""
    d = _linkx_mat(name, "snap_patents.mat")
    n = int(np.asarray(d["num_nodes"]).ravel()[0])
    ei = np.asarray(d["edge_index"])
    ei = ei if ei.shape[0] == 2 else ei.T
    years = np.asarray(d["years"], dtype=np.float64).ravel()
    X = d["node_feat"]
    del d
    labels = even_quantile_labels(years, 5)
    edges, _, _, report = canonicalize_edges(ei.T, n)
    del ei
    report |= {"source": "LINKX snap_patents.mat", "label_counts": np.bincount(labels[labels >= 0], minlength=5).tolist(),
               "unlabelled_nodes": int(np.sum(labels < 0)),
               "year_range_per_class": [[float(np.nanmin(years[labels == c])), float(np.nanmax(years[labels == c]))]
                                        for c in range(5)]}
    feats = _dense_features(X, report)
    del X
    splits = _optional_linkx_splits(name, labels, "snap-patents-splits.npy", report)
    splits += random_splits(labels, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, 5, splits, labels, features=feats,
                  meta={"source": "LINKX snap-patents (SNAP cit-Patents / NBER)", "extension": "a18",
                        "split_ids": [s.split_id for s in splits], "symmetrised_from_directed": True})
    return _finish(g, report)


def pokec(name: str = "pokec") -> GraphData:
    """pokec (Leskovec & Krevl, SNAP soc-Pokec; LINKX version, Lim et al. 2021): Slovak
    social network, label = gender, 65 profile features as built by LINKX;
    ``pokec.mat`` (keys ``edge_index``, ``node_feat``, ``num_nodes``, ``label``) from the
    authors' Google Drive (TO VERIFY). Raw SNAP soc-Pokec is registered as an
    alternative source but not used: LINKX's feature construction is not published, so
    it cannot be rebuilt identically. Directed friendships are symmetrised (§2.6);
    label -1 (missing gender, if any) is unlabelled. Splits: LINKX's 5 fixed splits when
    the registry lists ``aux_routes['splits']``, then 10 stratified random 60/20/20.
    Headline metric: accuracy (LINKX)."""
    d = _linkx_mat(name, "pokec.mat")
    n = int(np.asarray(d["num_nodes"]).ravel()[0])
    ei = np.asarray(d["edge_index"])
    ei = ei if ei.shape[0] == 2 else ei.T
    labels = np.asarray(d["label"]).ravel().astype(np.int64)
    X = d["node_feat"]
    del d
    if len(labels) != n:
        raise ValueError(f"pokec: {len(labels)} labels, num_nodes {n}")
    edges, _, _, report = canonicalize_edges(ei.T, n)
    del ei
    report |= {"source": "LINKX pokec.mat",
               "label_counts": {int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True), strict=True)}}
    feats = _dense_features(X, report)
    del X
    splits = _optional_linkx_splits(name, labels, "pokec-splits.npy", report)
    splits += random_splits(labels, 10, (0.6, 0.2, 0.2), seed=0)
    g = GraphData(name, n, edges, int(labels.max() + 1), splits, labels, features=feats,
                  meta={"source": "LINKX pokec (SNAP soc-Pokec)", "extension": "a18",
                        "split_ids": [s.split_id for s in splits], "symmetrised_from_directed": True})
    return _finish(g, report)


def _optional_linkx_splits(name: str, labels: np.ndarray, fname: str, report: dict) -> list[Split]:
    """LINKX fixed splits if the registry names a source for them (a static choice, so a
    graph's split set never depends on the network at load time)."""
    if not dataset_entry(name).get("aux_routes", {}).get("splits"):
        report["linkx_splits"] = "not registered (no stable source); random splits only"
        return []
    fixed, rep = _linkx_fixed_splits(_fetch_aux(name, "splits", fname, f"linkx/splits/{fname}"), labels, fname)
    report["linkx_splits"] = rep
    return fixed


LOADERS.update({
    **dict.fromkeys(GRAPHLAND, graphland),
    "ogbn-arxiv": ogbn_multiclass,
    "ogbn-products": ogbn_multiclass,
    "arxiv-year": arxiv_year,
    "twitch-gamers": twitch_gamers,
    "snap-patents": snap_patents,
    "pokec": pokec,
})
