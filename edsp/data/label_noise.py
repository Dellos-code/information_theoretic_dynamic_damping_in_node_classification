"""Train/validation label noise (Master Prompt §5.5; hypothesis H6 "finite T_L helps
under label noise").

The noise process is a data-generating process, not a method: it reads the true labels
through ``LabelVault.descriptive_labels()`` (this module is on the descriptive allow-list)
and returns an *observed* label array that differs from the true one only on the train
and validation nodes of the split. Test labels are never altered. Methods and model
selection then see the observed labels through ``train_labels()`` / ``val_labels()``, so
both training and validation are noisy, as they would be in practice; evaluation is on
clean test labels.

Specs (rate r in [0, 1]):
  ``uniform:r``   each train/val node keeps its label with probability 1 - r; otherwise
                  the label is replaced by one of the other k - 1 classes, uniformly.
  ``pairwise:r``  with probability r the label c becomes (c + 1) mod k (a fixed confusion
                  pair per class). For k = 2 both specs coincide.
Randomness: ``numpy.random.default_rng([seed, crc32(split_id)])``; one uniform draw and one
class draw per train/val node regardless of the outcome, so the stream does not depend on r.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from .vault import LabelVault

MODES = ("uniform", "pairwise")


def parse(spec: str) -> tuple[str, float]:
    m = re.fullmatch(r"(uniform|pairwise):([0-9]*\.?[0-9]+)", spec.strip())
    if not m:
        raise ValueError(f"label-noise spec must be 'uniform:r' or 'pairwise:r', got {spec!r}")
    rate = float(m.group(2))
    if not 0.0 <= rate <= 1.0:
        raise ValueError("label-noise rate must lie in [0, 1]")
    return m.group(1), rate


def tag(spec: str) -> str:
    mode, rate = parse(spec)
    return f"ln-{mode}-{rate:g}"


def observe(vault: LabelVault, spec: str, seed: int) -> tuple[np.ndarray, dict]:
    mode, rate = parse(spec)
    y = vault.descriptive_labels()
    if y.ndim != 1:
        raise NotImplementedError("label noise is defined for single-label graphs")
    k = int(vault.num_classes)
    tr, va = np.asarray(vault.train_idx), np.asarray(vault.val_idx)
    idx = np.concatenate([tr, va])
    rng = np.random.default_rng([int(seed), zlib.crc32(str(vault.split_id).encode())])
    draw = rng.random(len(idx))
    shift = rng.integers(1, max(k, 2), size=len(idx))
    valid = y[idx] >= 0
    flip = (draw < rate) & valid & (k > 1)
    new = (y[idx] + (shift if mode == "uniform" else 1)) % k
    observed = y.copy()
    observed[idx[flip]] = new[flip]
    info = {"spec": spec, "seed": int(seed), "mode": mode, "rate": rate,
            "train": len(tr), "val": len(va),
            "flipped_train": int(flip[: len(tr)].sum()), "flipped_val": int(flip[len(tr):].sum())}
    return observed, info
