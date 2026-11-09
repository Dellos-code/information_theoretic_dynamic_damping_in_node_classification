"""Label isolation (Master Prompt §2.7, rule R3).

`LabelVault` is the only object that holds node labels after loading. Access is
checked against the *calling module*:

* ``train_labels()``      any caller.
* ``val_labels()``        only modules under ``edsp.selection`` or ``edsp.eval``.
* ``test_labels()``       only ``edsp.eval.evaluate``.
* ``descriptive_labels()`` only ``edsp.data.stats``, ``edsp.data.perturb`` and
  ``edsp.data.label_noise``; used for descriptive dataset statistics (Master Prompt
  §3.2 "full-label descriptive version") and as the data-generating process of the
  PERTURBED-REAL sweeps (edge re-wiring; train/val label noise, §5.5), never by a
  method. Every such access is logged and flagged.

Label noise (§5.5, H6): a vault may carry an *observed* label array. ``train_labels()``
and ``val_labels()`` then return the observed (possibly corrupted) labels, while
``test_labels()`` and ``descriptive_labels()`` always return the true labels; the
observed array must agree with the true labels on every test node.
* ``oracle_labels()``     only ``edsp.regimes.oracle`` *and* only when the
  environment variable ``ORACLE=1`` is set (H-ORACLE upper bound, §2.3g).

Python cannot make the stored array truly private; the guarantee is enforced by
the caller check, by the access log and by the T11 leakage test, which runs
methods in strict mode and asserts that no test-label access happened outside
``edsp.eval.evaluate``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_VAL_PREFIXES = ("edsp.selection", "edsp.eval")
_TEST_MODULES = ("edsp.eval.evaluate",)
_DESCRIPTIVE_MODULES = ("edsp.data.stats", "edsp.data.perturb", "edsp.data.label_noise")
_ORACLE_MODULES = ("edsp.regimes.oracle",)


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
    _observed: np.ndarray | None = field(default=None, repr=False)  # noisy train/val labels (§5.5)
    noise_info: dict | None = None

    def __post_init__(self) -> None:
        self._labels = np.asarray(self._labels)
        self._labels.setflags(write=False)
        if self._observed is not None:
            obs = np.array(self._observed, copy=True)
            if obs.shape != self._labels.shape:
                raise ValueError("observed labels must have the shape of the true labels")
            if not np.array_equal(obs[np.asarray(self.test_idx, dtype=np.int64)],
                                  self._labels[np.asarray(self.test_idx, dtype=np.int64)]):
                raise ValueError("observed labels must equal the true labels on test nodes")
            obs.setflags(write=False)
            self._observed = obs
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
        src = self._labels if self._observed is None else self._observed
        return self.train_idx.copy(), src[self.train_idx].copy()

    def val_labels(self) -> tuple[np.ndarray, np.ndarray]:
        self._check("val", _VAL_PREFIXES)
        src = self._labels if self._observed is None else self._observed
        return self.val_idx.copy(), src[self.val_idx].copy()

    def test_labels(self) -> tuple[np.ndarray, np.ndarray]:
        self._check("test", _TEST_MODULES)
        return self.test_idx.copy(), self._labels[self.test_idx].copy()

    def descriptive_labels(self) -> np.ndarray:
        self._check("descriptive", _DESCRIPTIVE_MODULES)
        return self._labels.copy()

    def oracle_labels(self) -> np.ndarray:
        if os.environ.get("ORACLE") != "1":
            caller = self._caller(2)
            self.access_log.append(AccessRecord("oracle", caller, False))
            raise LabelAccessError("oracle labels require ORACLE=1 (scripts/oracle_upper_bound.py)")
        self._check("oracle", _ORACLE_MODULES)
        return self._labels.copy()

    def oracle_accessed(self) -> bool:
        return any(r.kind == "oracle" for r in self.access_log)

    def test_accesses_outside_evaluate(self) -> list[AccessRecord]:
        return [r for r in self.access_log if r.kind in {"test", "descriptive", "oracle"} and not r.allowed]
