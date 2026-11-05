"""Label isolation (Master Prompt §2.7, rule R3).

`LabelVault` is the only object that holds node labels after loading. Access is
checked against the *calling module*:

* ``train_labels()``      any caller.
* ``val_labels()``        only modules under ``edsp.selection`` or ``edsp.eval``.
* ``test_labels()``       only ``edsp.eval.evaluate``.
* ``descriptive_labels()`` only ``edsp.data.stats``; used for descriptive dataset
  statistics (Master Prompt §3.2 "full-label descriptive version"), never by a
  method. Every such access is logged and flagged.

Python cannot make the stored array truly private; the guarantee is enforced by
the caller check, by the access log and by the T11 leakage test, which runs
methods in strict mode and asserts that no test-label access happened outside
``edsp.eval.evaluate``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

_VAL_PREFIXES = ("edsp.selection", "edsp.eval")
_TEST_MODULES = ("edsp.eval.evaluate",)
_DESCRIPTIVE_MODULES = ("edsp.data.stats",)


class LabelAccessError(PermissionError):
    """Raised when a module asks for labels it is not allowed to see."""


@dataclass(frozen=True)
class AccessRecord:
    kind: str
    caller: str
    allowed: bool


@dataclass
class LabelVault:
    """Holds labels for one split. Labels are int class ids (n,) or binary (n, tasks)."""

    _labels: np.ndarray = field(repr=False)
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    num_classes: int
    split_id: str = "0"
    strict: bool = True
    access_log: list[AccessRecord] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._labels = np.asarray(self._labels)
        self._labels.setflags(write=False)
        for name in ("train_idx", "val_idx", "test_idx"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.int64))
        overlap = (
            np.intersect1d(self.train_idx, self.val_idx).size
            + np.intersect1d(self.train_idx, self.test_idx).size
            + np.intersect1d(self.val_idx, self.test_idx).size
        )
        if overlap:
            raise ValueError("train/val/test indices overlap")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _caller(depth: int = 2) -> str:
        frame = sys._getframe(depth)
        return str(frame.f_globals.get("__name__", "<unknown>"))

    def _check(self, kind: str, allowed_prefixes: tuple[str, ...] | None) -> None:
        caller = self._caller(3)
        allowed = allowed_prefixes is None or any(
            caller == p or caller.startswith(p + ".") for p in allowed_prefixes
        )
        self.access_log.append(AccessRecord(kind, caller, allowed))
        if not allowed and self.strict:
            raise LabelAccessError(f"{caller} may not read {kind} labels")

    # ------------------------------------------------------------------ public
    def train_labels(self) -> tuple[np.ndarray, np.ndarray]:
        self._check("train", None)
        return self.train_idx.copy(), self._labels[self.train_idx].copy()

    def val_labels(self) -> tuple[np.ndarray, np.ndarray]:
        self._check("val", _VAL_PREFIXES)
        return self.val_idx.copy(), self._labels[self.val_idx].copy()

    def test_labels(self) -> tuple[np.ndarray, np.ndarray]:
        self._check("test", _TEST_MODULES)
        return self.test_idx.copy(), self._labels[self.test_idx].copy()

    def descriptive_labels(self) -> np.ndarray:
        self._check("descriptive", _DESCRIPTIVE_MODULES)
        return self._labels.copy()

    def test_accesses_outside_evaluate(self) -> list[AccessRecord]:
        return [r for r in self.access_log if r.kind in {"test", "descriptive"} and not r.allowed]
