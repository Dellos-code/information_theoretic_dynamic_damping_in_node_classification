import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

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
    assert np.isclose(kl_to_uniform(p), math.log(len(p)) - entropy(p)[0])


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


def test_t9_backend_reference_is_deterministic():
    kwargs = dict(edges=[(0, 1), (1, 2)], probabilities=[0.2, 0.7], heterophily=[0, 1],
                  priors=[[1, 0], [0.5, 0.5], [0, 1]], max_iter=3)
    assert np.array_equal(edsp(**kwargs).probabilities, edsp(**kwargs).probabilities)


def test_t9_torch_adapter_parity_when_available():
    torch = pytest.importorskip("torch")
    from edsp import edsp_torch

    kwargs = dict(edges=[(0, 1)], probabilities=[0.5], heterophily=[1],
                  priors=[[1, 0], [0.5, 0.5]], max_iter=2)
    result = edsp(**kwargs)
    torch_result = edsp_torch(**kwargs)
    assert torch.allclose(torch_result["probabilities"], torch.from_numpy(result.probabilities), atol=1e-10)


def test_t10_degenerate_nodes():
    result = edsp([], [], [], [[1, 0], [0.5, 0.5]], max_iter=1)
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
