"""Model selection: may read validation labels (rule R3), never test labels."""

from __future__ import annotations

import numpy as np

from edsp.data.vault import LabelVault


def validation_accuracy(vault: LabelVault, probabilities: np.ndarray) -> float:
    idx, y = vault.val_labels()
    return float(np.mean(np.asarray(probabilities)[idx].argmax(axis=1) == y))
