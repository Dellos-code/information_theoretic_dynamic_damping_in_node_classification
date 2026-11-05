"""Canonical in-memory graph format used by every loader."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .vault import LabelVault


@dataclass(frozen=True)
class Split:
    split_id: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    source: str  # "official" | "random-60/20/20-seed<k>" | ...


@dataclass
class GraphData:
    """Undirected simple graph plus optional edge information.

    ``edges`` holds each undirected edge once, as (i, j) with i < j, without
    self-loops or duplicates. ``edge_attr`` (m, d) and ``edge_sign`` (m,) are
    aligned with ``edges``. Labels live only inside the vaults returned by
    :meth:`vault`.
    """

    name: str
    num_nodes: int
    edges: np.ndarray
    num_classes: int
    splits: list[Split]
    _labels: np.ndarray = field(repr=False)
    features: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    edge_sign: np.ndarray | None = None
    edge_relation: np.ndarray | None = None  # (m,) int relation id, for H-RELATION (§2.3f)
    multilabel: bool = False
    meta: dict = field(default_factory=dict)

    def vault(self, split: int | str = 0, *, strict: bool = True) -> LabelVault:
        chosen = self.splits[split] if isinstance(split, int) else next(
            s for s in self.splits if s.split_id == split
        )
        return LabelVault(
            self._labels,
            chosen.train_idx,
            chosen.val_idx,
            chosen.test_idx,
            num_classes=self.num_classes,
            split_id=chosen.split_id,
            strict=strict,
        )

    def descriptive_vault(self) -> LabelVault:
        """Vault whose only legitimate use is edsp.data.stats (all nodes in 'train')."""
        return LabelVault(
            self._labels,
            np.arange(self.num_nodes),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            num_classes=self.num_classes,
            split_id="descriptive",
        )

    def content_hash(self) -> str:
        """SHA-256 of the processed tensors (rule R1 'dataset hash')."""
        h = hashlib.sha256()
        for arr in (self.edges, self._labels, self.features, self.edge_attr, self.edge_sign):
            if arr is not None:
                a = np.ascontiguousarray(arr)
                h.update(str(a.dtype).encode() + str(a.shape).encode())
                h.update(a.tobytes())
        return "sha256:" + h.hexdigest()


def canonicalize_edges(
    edges: np.ndarray,
    num_nodes: int,
    edge_attr: np.ndarray | None = None,
    edge_sign: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    """Symmetrise, drop self-loops and merge duplicates.

    Duplicate (or reciprocal) rows are merged; for ``edge_attr`` the first
    occurrence is kept unless all occurrences agree, and the number of
    conflicting merges is reported. ``edge_sign`` is summed over occurrences,
    as specified for H-NATIVE in Master Prompt §2.3(a); conflicting reciprocal
    signs are counted.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    report: dict = {"input_rows": len(edges)}
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge index out of range")
    loops = edges[:, 0] == edges[:, 1]
    report["self_loops_removed"] = int(loops.sum())
    keep = ~loops
    edges = edges[keep]
    attr = None if edge_attr is None else np.asarray(edge_attr)[keep]
    sign = None if edge_sign is None else np.asarray(edge_sign, dtype=np.float64)[keep]
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    key = lo * num_nodes + hi
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    first = np.ones(len(key_sorted), dtype=bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    group = np.cumsum(first) - 1
    uniq = order[first]
    out = np.stack([lo[uniq], hi[uniq]], axis=1)
    report["duplicates_merged"] = int(len(edges) - len(out))
    out_attr = None
    if attr is not None:
        out_attr = attr[uniq]
        neq = attr[order] != out_attr[group]
        differs = neq.reshape(len(neq), -1).any(axis=1)
        report["conflicting_attr_merges"] = int(np.unique(group[differs]).size)
    out_sign = None
    if sign is not None:
        out_sign = np.bincount(group, weights=sign[order], minlength=len(out))
        pos = np.bincount(group, weights=(sign[order] > 0).astype(float), minlength=len(out))
        neg = np.bincount(group, weights=(sign[order] < 0).astype(float), minlength=len(out))
        report["sign_conflicting_pairs"] = int(np.sum((pos > 0) & (neg > 0)))
    report["undirected_edges"] = len(out)
    return out, out_attr, out_sign, report


def random_splits(
    labels: np.ndarray,
    n_splits: int = 10,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> list[Split]:
    """Stratified random splits (used only where no official split exists)."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("random_splits expects single-label data")
    out = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        tr, va, te = [], [], []
        for c in np.unique(labels[labels >= 0]):
            idx = rng.permutation(np.flatnonzero(labels == c))
            a = round(fractions[0] * len(idx))
            b = a + round(fractions[1] * len(idx))
            tr.append(idx[:a])
            va.append(idx[a:b])
            te.append(idx[b:])
        out.append(
            Split(
                f"rand{s}",
                np.sort(np.concatenate(tr)),
                np.sort(np.concatenate(va)),
                np.sort(np.concatenate(te)),
                f"random-stratified-{fractions[0]:.0%}/{fractions[1]:.0%}/{fractions[2]:.0%}-seed{seed + s}",
            )
        )
    return out
