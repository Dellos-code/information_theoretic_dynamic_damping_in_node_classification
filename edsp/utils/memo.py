"""Opt-in reuse of label-free structural quantities across runs on the same graph.

Suite C4 runs every method on 112 binary task views of ogbn-proteins that share one
edge array (``GraphData.task_view``). Quantities that depend only on the edges and edge
weights — the sparse adjacency, its spectral radius, its connected components, the
structural node embedding of H-LEARNED (Track A) — are identical for all tasks and
all hyper-parameter values, so recomputing them per call only costs time.

``shared_structure()`` opens a scope in which :func:`memo` returns the value computed
on the first call for the same tag and the same argument *objects* (identity, not
equality; the scope keeps a reference to each key object, so an id cannot be reused
while the scope is open). Outside the scope :func:`memo` always recomputes, so
behaviour without the scope is unchanged. Only label-free functions may use it:
their inputs are edges and edge weights, never a vault.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_ACTIVE: dict | None = None


@contextmanager
def shared_structure() -> Iterator[dict]:
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, {}
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = previous


def memo(tag: str, keys: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
    if _ACTIVE is None:
        return compute()
    slot = (tag, *(id(k) for k in keys))
    hit = _ACTIVE.get(slot)
    if hit is not None and all(a is b for a, b in zip(hit[0], keys, strict=True)):
        return hit[1]
    value = compute()
    _ACTIVE[slot] = (keys, value)
    return value
