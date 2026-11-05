"""PR-7 tests: §5.4 metrics and §5.9 statistics."""

import math

import numpy as np

from edsp.eval.metrics import average_precision, classification_metrics
from edsp.eval.stats_tests import friedman_nemenyi, holm, wilcoxon_paired


def test_perfect_and_uniform_predictions():
    y = np.array([0, 1, 2, 1])
    m = classification_metrics(y, np.eye(3)[y], 3)
    assert m["accuracy"] == m["macro_f1"] == m["balanced_accuracy"] == 1.0
    assert m["brier"] == 0.0 and m["ece"] == 0.0 and m["nll"] < 1e-9
    u = classification_metrics(y, np.full((4, 3), 1 / 3), 3)
    assert math.isclose(u["nll"], math.log(3))
    assert math.isclose(u["brier"], 2 / 3)


def test_binary_auc_ap_and_ece():
    y = np.array([0, 0, 1, 1])
    P = np.array([[0.9, 0.1], [0.6, 0.4], [0.35, 0.65], [0.2, 0.8]])
    m = classification_metrics(y, P, 2)
    assert m["roc_auc"] == 1.0 and m["average_precision"] == 1.0
    assert math.isclose(average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1])), (1 + 2 / 3) / 2)
    # all confident and all correct -> ECE = 1 - mean confidence
    assert math.isclose(m["ece"], np.mean(1 - P.max(1)))


def test_holm_and_wilcoxon():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert math.isnan(wilcoxon_paired(np.ones(5), np.ones(5)))
    assert wilcoxon_paired(np.arange(10) + 1.0, np.zeros(10)) < 0.01


def test_friedman_nemenyi_ranks_and_cd():
    S = np.array([[0.9, 0.8, 0.7], [0.85, 0.8, 0.6], [0.95, 0.7, 0.65], [0.9, 0.85, 0.5]])
    r = friedman_nemenyi(S)
    assert np.allclose(r["average_ranks"], [1, 2, 3])
    # Demsar (2006): q_0.05 for k = 3 is 2.343
    assert math.isclose(r["nemenyi_cd"], 2.343 * math.sqrt(3 * 4 / (6 * 4)), rel_tol=1e-3)
