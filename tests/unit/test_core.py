import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from edsp import edsp, entropy, heterophily_complement, kl_to_uniform, trust_calibrate


@st.composite
def distributions(draw, min_k=2, max_k=20):
    k = draw(st.integers(min_value=min_k, max_value=max_k))
    values = draw(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False), min_size=k, max_size=k))
    values = np.asarray(values, dtype=np.float64)
    if values.sum() == 0:
        values[0] = 1
    return values / values.sum()


@settings(max_examples=500)
@given(distributions())
def test_t1_complement_preserves_simplex(p):
    result = heterophily_complement(p)
    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1)


@settings(max_examples=500)
@given(distributions())
def test_t2_complement_involution_scaling(p):
    k = len(p)
    u = np.full(k, 1 / k)
    assert np.allclose(heterophily_complement(heterophily_complement(p)), u + (p - u) / (k - 1) ** 2)


@settings(max_examples=500)
@given(distributions())
def test_t3_kl_identity(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.sum(np.where(p > 0, p * np.log(len(p) * p), 0.0))
    assert np.isclose(kl_to_uniform(p), direct)


@settings(max_examples=500)
@given(distributions())
def test_t4_entropy_base_invariance(p):
    assert np.isclose(entropy(p, base=2), entropy(p) / math.log(2))
    assert np.isclose(entropy(p, base=10), entropy(p) / math.log(10))


def test_t5_regimes_and_toy_examples():
    assert np.allclose(heterophily_complement([0.7, 0.3]), [0.3, 0.7], atol=1e-12)
    assert np.allclose(heterophily_complement([0.5, 0.25, 0.25]), [0.25, 0.375, 0.375], atol=1e-12)
    result = edsp([(0, 1)], [0.5], [0], [[1, 0], [0, 1]], max_iter=1)
    assert np.allclose(result.probabilities[0], [2 / 3, 1 / 3], atol=1e-12)
    result = edsp([(0, 1)], [0.5], [1], [[1, 0], [0, 1]], max_iter=1)
    assert np.allclose(result.probabilities[0], [1, 0], atol=1e-12)
    result = edsp([(0, 1)], [0.5], [0], [[0.5, 0.5], [0, 1]], max_iter=1)
    assert np.allclose(result.probabilities[0], result.probabilities[1], atol=1e-12)
    for p in (0.1, 0.5, 0.9):
        result = edsp([(0, 1)], [p], [0], [[0.5, 0.5], [0, 1]], max_iter=1)
        assert np.allclose(result.probabilities[0], result.probabilities[1])


def test_p1_alpha_ratio():
    result = edsp([(0, 1)], [0.4], [0], [[0.8, 0.2], [0.2, 0.8]], max_iter=1)
    alpha = result.last_alpha[0]
    lhs = alpha / (1 - alpha)
    rhs = kl_to_uniform([[0.2, 0.8]])[0] / kl_to_uniform([[0.8, 0.2]])[0] * result.p_esc[0]
    assert np.isclose(lhs, rhs)


def test_d1_one_hot_alpha_bound():
    for p_esc in (0.01, 0.2, 0.9, 1.0):
        p = 1 - (1 - p_esc)
        result = edsp([(0, 1)], [p], [0], [[1, 0], [0, 1]], max_iter=1)
        assert np.all(result.last_alpha <= 0.5 + 1e-12)


def test_t6_calibration_kl_monotone():
    trusts = [0.0, 0.5, 1.0, 2.0, np.inf]
    values = [kl_to_uniform(trust_calibrate([[1, 0]], [[0.5]], trust, np.inf)[0])[0] for trust in trusts]
    assert np.all(np.diff(values) >= -1e-12)


def test_t6_trust():
    v, p, _ = trust_calibrate([[1, 0]], [0.8], 0, 0)
    assert np.allclose(v, [[0.5, 0.5]])
    assert p[0] == 0
    v_inf, p_inf, _ = trust_calibrate([[0.8, 0.2]], [0.8], np.inf, np.inf)
    assert np.allclose(v_inf, [[0.8, 0.2]]) and np.allclose(p_inf, [0.8])


def test_t7_weights_and_escape_probability():
    result = edsp([(0, 1), (0, 2)], [0.2, 0.5], [0, 0], np.full((3, 2), 0.5), max_iter=1)
    assert np.isclose(result.weights[0] + result.weights[1], 1)
    assert np.isclose(result.p_esc[0], 1 - (1 - 0.2) * (1 - 0.5))


@settings(max_examples=500)
@given(distributions(), st.integers(2, 20))
def test_t8_iterates_stay_in_simplex(prior, iterations):
    result = edsp([(0, 1)], [0.5], [0], np.vstack([prior, prior]), max_iter=iterations)
    assert np.all(result.probabilities >= 0)
    assert np.allclose(result.probabilities.sum(axis=1), 1)


def test_t8_random_graphs():
    rng = np.random.default_rng(7)
    for k in (2, 5, 20):
        n = 50
        edges = rng.integers(0, n, size=(120, 2))
        edges = edges[edges[:, 0] != edges[:, 1]]
        result = edsp(edges, rng.uniform(0.01, 0.9, len(edges)), rng.integers(0, 2, len(edges)),
                      rng.dirichlet(np.ones(k), size=n), max_iter=4)
        assert np.all(result.probabilities >= 0)
        assert np.allclose(result.probabilities.sum(axis=1), 1)


def test_t9_backend_reference_is_deterministic():
    kwargs = {"edges": [(0, 1), (1, 2)], "probabilities": [0.2, 0.7], "heterophily": [0, 1],
                  "priors": [[1, 0], [0.5, 0.5], [0, 1]], "max_iter": 3}
    assert np.array_equal(edsp(**kwargs).probabilities, edsp(**kwargs).probabilities)


def test_t9_torch_adapter_parity_when_available():
    import torch

    from edsp import edsp_torch

    rng = np.random.default_rng(42)
    for k in (2, 5, 20):
        n = 55
        edges = rng.integers(0, n, size=(100, 2))
        edges = edges[edges[:, 0] != edges[:, 1]]
        probabilities = rng.uniform(0.01, 0.9, len(edges))
        heterophily = rng.integers(0, 2, len(edges))
        priors = rng.dirichlet(np.ones(k), size=n)
        priors[0] = np.eye(k)[0]
        kwargs = {
            "edges": edges,
            "probabilities": probabilities,
            "heterophily": heterophily,
            "priors": priors,
            "max_iter": 4,
        }
        result = edsp(**kwargs)
        torch_result = edsp_torch(**kwargs)
        assert torch.allclose(
            torch_result["probabilities"], torch.from_numpy(result.probabilities), atol=1e-10
        )


def test_t10_degenerate_nodes():
    result = edsp([(0, 1)], [0.0], [0], [[1, 0], [0.5, 0.5]], max_iter=1)
    assert np.allclose(result.probabilities, [[1, 0], [0.5, 0.5]])


@pytest.mark.xfail(reason="Deferred to PR-2 LabelVault and PR-4 train-only regimes", strict=True)
def test_t11_leakage_guard():
    raise NotImplementedError("LabelVault is intentionally deferred")


def test_e6_two_node_heterophily():
    result = edsp([(0, 1)], [0.9], [1], [[1, 0], [0.5, 0.5]], max_iter=1)
    assert np.allclose(result.probabilities[1], [0, 1], atol=1e-12)


def test_e7_cycle_runs_without_hidden_damping():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    result = edsp(edges, [0.9] * 4, [1] * 4, [[0.9, 0.1]] * 4, max_iter=20)
    assert result.iterations == 20
    assert result.oscillation and result.oscillation_period == 2
    assert result.oscillation_amplitude > 0


@pytest.mark.parametrize("p, expected_converged", [(0.5, True), (0.9, False)])
def test_oscillation_one_homophilic_edge(p, expected_converged):
    result = edsp([(0, 1)], [p], [0], [[0.9, 0.1], [0.1, 0.9]])
    assert result.converged is expected_converged
    assert result.oscillation is (p == 0.9)
    if p == 0.5:
        assert 250 <= result.iterations <= 270


@pytest.mark.parametrize("p", [0.5, 0.7, 0.9])
def test_oscillation_four_cycle(p):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    result = edsp(edges, [p] * 4, [1] * 4, [[0.9, 0.1]] * 4)
    assert result.oscillation


def test_one_hot_priors_cycle():
    result = edsp([(0, 1)], [0.9], [0], [[1, 0], [0, 1]])
    assert result.oscillation


def test_diagnostics_are_bounded():
    result = edsp([(0, 1)], [0.9], [0], [[0.9, 0.1], [0.1, 0.9]], max_iter=100)
    assert sorted(result.alpha_histograms) == [1, 5, 100]
    assert len(result.regime_trace) == result.iterations
    assert len(result.entropy_trace) == result.iterations
    assert not hasattr(result, "delta_trace")
