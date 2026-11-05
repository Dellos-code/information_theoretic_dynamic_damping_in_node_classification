import importlib.util
import math
import os

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from edsp import (
    canonical_phi,
    edsp,
    edsp_torch,
    entropy,
    heterophily_complement,
    kl_to_uniform,
    trust_calibrate,
)


@st.composite
def distributions(draw, min_k=2, max_k=20):
    k = draw(st.integers(min_value=min_k, max_value=max_k))
    values = draw(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False), min_size=k, max_size=k))
    values = np.asarray(values, dtype=np.float64)
    if values.sum() == 0:
        values[0] = 1
    return values / values.sum()


@st.composite
def paired_distributions(draw, min_k=2, max_k=20):
    k = draw(st.integers(min_value=min_k, max_value=max_k))
    first = np.asarray(
        draw(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False), min_size=k, max_size=k)),
        dtype=np.float64,
    )
    second = np.asarray(
        draw(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False), min_size=k, max_size=k)),
        dtype=np.float64,
    )
    if first.sum() == 0:
        first[0] = 1
    if second.sum() == 0:
        second[0] = 1
    return first / first.sum(), second / second.sum()


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
@given(paired_distributions())
def test_p1_alpha_ratio_property(pair):
    vj, neighbor = pair
    assume(kl_to_uniform(vj) > 1e-10)
    result = edsp([(0, 1)], [0.4], [0], np.vstack([vj, neighbor]), max_iter=1)
    alpha = result.last_alpha[0]
    assume(alpha < 1 - 1e-10)
    lhs = alpha / (1 - alpha)
    rhs = kl_to_uniform(neighbor) / kl_to_uniform(vj) * result.p_esc[0]
    assert np.isclose(lhs, rhs, atol=1e-10, rtol=1e-10)


@settings(max_examples=500)
@given(distributions())
def test_t4_entropy_base_invariance(p):
    assert np.isclose(entropy(p, base=2), entropy(p) / math.log(2))
    assert np.isclose(entropy(p, base=10), entropy(p) / math.log(10))


@settings(max_examples=500)
@given(
    st.integers(min_value=2, max_value=20),
    st.floats(min_value=1e-6, max_value=0.999999),
    st.data(),
)
def test_d1_one_hot_alpha_bound_property(k, p, neighbor):
    neighbor = np.asarray(
        neighbor.draw(
            st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False), min_size=k, max_size=k),
            label="neighbor",
        ),
        dtype=np.float64,
    )
    if neighbor.sum() == 0:
        neighbor[0] = 1
    neighbor = neighbor / neighbor.sum()
    one_hot = np.eye(k)[0]
    result = edsp([(0, 1)], [p], [0], np.vstack([one_hot, neighbor]), max_iter=1)
    assert result.last_alpha[0] <= 0.5 + 1e-12


def test_t5_regimes_and_toy_examples():
    assert np.allclose(heterophily_complement([0.7, 0.3]), [0.3, 0.7], atol=1e-12)
    assert np.allclose(heterophily_complement([0.5, 0.25, 0.25]), [0.25, 0.375, 0.375], atol=1e-12)
    result = edsp([(0, 1)], [0.5], [0], [[1, 0], [0, 1]], max_iter=1)
    assert np.allclose(result.probabilities[0], [2 / 3, 1 / 3], atol=1e-12)
    result = edsp([(0, 1)], [0.5], [1], [[1, 0], [0, 1]], max_iter=1)
    assert np.allclose(result.probabilities[0], [1, 0], atol=1e-12)
    result = edsp([(0, 1)], [0.5], [0], [[0.5, 0.5], [0, 1]], max_iter=1)
    assert np.allclose(result.probabilities[0], result.probabilities[1], atol=1e-12)


def test_t5_regime1_explicit():
    result = edsp([], [], [], [[0.5, 0.5]], max_iter=1)
    assert np.allclose(result.regime_trace[0], [1.0, 0.0, 0.0, 0.0])


def test_t5_regime3_explicit():
    result = edsp([], [], [], [[1.0, 0.0]], max_iter=1)
    assert np.allclose(result.regime_trace[0], [0.0, 0.0, 1.0, 0.0])


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
    if importlib.util.find_spec("torch") is None:
        if os.getenv("EDSP_REQUIRE_TORCH") == "1":
            pytest.fail("EDSP_REQUIRE_TORCH=1 but torch is not installed")
        pytest.skip("torch is unavailable in this environment")
    import torch

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


def _legacy_gs(edges, probabilities, heterophily, priors, *, order="id", seed=0, max_iter=4):
    priors = np.asarray(priors, dtype=np.float64)
    v = priors / priors.sum(axis=1, keepdims=True)
    n, k = v.shape
    edges = np.asarray(edges, dtype=np.int64)
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    p = np.concatenate((np.asarray(probabilities), np.asarray(probabilities)))
    h = np.concatenate((np.asarray(heterophily), np.asarray(heterophily)))
    order_idx = np.argsort(directed[:, 1], kind="stable")
    directed, p, h = directed[order_idx], p[order_idx], h[order_idx]
    source, target = directed[:, 0], directed[:, 1]
    log_odds = -np.log1p(-p)
    den = np.bincount(target, weights=log_odds, minlength=n)
    weights = np.divide(log_odds, den[target], out=np.zeros_like(log_odds), where=den[target] > 0)
    p_esc = -np.expm1(-den)
    base_entropy = entropy(v)
    hmax = math.log(k)
    rng = np.random.default_rng(seed)
    current = v.copy()
    for _ in range(max_iter):
        update_order = np.arange(n) if order == "id" else rng.permutation(n)
        next_values = current.copy()
        for node in update_order:
            incoming = np.flatnonzero(target == node)
            if incoming.size:
                transformed = np.where(h[incoming, None] == 0, next_values[source[incoming]],
                                       (1.0 - next_values[source[incoming]]) / (k - 1))
                aggregate = np.sum(weights[incoming, None] * transformed, axis=0)
            else:
                aggregate = np.full(k, 1.0 / k)
            h_prime = float(entropy(aggregate)[0])
            numerator = (hmax - h_prime) * p_esc[node]
            denominator = numerator + (hmax - base_entropy[node])
            alpha = 0.0 if denominator <= 1e-15 else numerator / denominator
            boundary = (
                (abs(hmax - base_entropy[node]) <= 1e-12 * hmax and abs(hmax - h_prime) <= 1e-12 * hmax)
                or (p_esc[node] == 0 and abs(hmax - base_entropy[node]) <= 1e-12 * hmax)
            )
            value = (1 - alpha) * v[node] + alpha * aggregate
            next_values[node] = np.full(k, 1.0 / k) if boundary else value / value.sum()
        current = next_values
    return current


def test_edsp_gs_csr_offsets_equivalent_to_previous_impl():
    rng = np.random.default_rng(23)
    n = 40
    k = 4
    edges = rng.integers(0, n, size=(120, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    probabilities = rng.uniform(0.01, 0.95, len(edges))
    heterophily = rng.integers(0, 2, len(edges))
    priors = rng.dirichlet(np.ones(k), size=n)
    expected = _legacy_gs(edges, probabilities, heterophily, priors, order="random", seed=5, max_iter=6)
    got = edsp(edges, probabilities, heterophily, priors, variant="EDSP-GS", order="random", seed=5, max_iter=6)
    assert np.allclose(got.probabilities, expected, atol=1e-12)


def test_canonical_rejects_beta_not_one():
    with pytest.raises(ValueError, match="EDSP-canonical requires beta=1"):
        edsp([(0, 1)], [0.5], [0], [[0.8, 0.2], [0.2, 0.8]], variant="EDSP-canonical", beta=0.5)


def test_t10_degenerate_nodes():
    result = edsp([(0, 1)], [0.0], [0], [[1, 0], [0.5, 0.5]], max_iter=1)
    assert np.allclose(result.probabilities, [[1, 0], [0.5, 0.5]])


@pytest.mark.xfail(reason="LabelVault exists since PR-2; T11 needs the method regimes of PR-4 to run every method in strict mode", strict=True)
def test_t11_leakage_guard():
    raise NotImplementedError("LabelVault is intentionally deferred")


def test_e6_two_node_heterophily():
    result = edsp([(0, 1)], [0.9], [1], [[1, 0], [0.5, 0.5]], max_iter=1)
    assert np.allclose(result.probabilities[1], [0, 1], atol=1e-12)


def test_e7_unlabelled_4cycle_one_nonuniform_start():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    priors = np.array([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    jacobi = edsp(edges, [0.9] * 4, [1] * 4, priors, max_iter=1000)
    gs = edsp(edges, [0.9] * 4, [1] * 4, priors, variant="EDSP-GS", max_iter=1000)
    assert jacobi.iterations > 0
    assert gs.converged


def test_case_b_p03_converges():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    priors = np.array([[0.9, 0.1]] * 4)
    result = edsp(edges, [0.3] * 4, [1] * 4, priors, max_iter=1000)
    assert result.converged
    assert not result.oscillation


def test_fixed_point_multiple_solutions_case_a_p09():
    edges = [(0, 1)]
    priors = np.array([[0.9, 0.1], [0.1, 0.9]])
    gs = edsp(edges, [0.9], [0], priors, variant="EDSP-GS", order="id")
    damped = edsp(edges, [0.9], [0], priors, variant="EDSP-damped", beta=0.5)
    assert np.allclose(gs.probabilities[0], [0.5228, 0.4772], atol=3e-4)
    assert np.allclose(gs.probabilities[1], [0.1011, 0.8989], atol=3e-4)
    assert np.allclose(damped.probabilities[0], [0.7466, 0.2534], atol=3e-4)
    assert np.allclose(damped.probabilities[1], [0.2534, 0.7466], atol=3e-4)
    gs_phi = canonical_phi(edges, [0.9], [0], priors, gs.probabilities)
    damped_phi = canonical_phi(edges, [0.9], [0], priors, damped.probabilities)
    assert np.max(np.sum(np.abs(gs_phi - gs.probabilities), axis=1)) < 1e-7
    assert np.max(np.sum(np.abs(damped_phi - damped.probabilities), axis=1)) < 1e-7


def test_diagnostics_are_bounded():
    result = edsp([(0, 1)], [0.9], [0], [[0.9, 0.1], [0.1, 0.9]], max_iter=100)
    assert sorted(result.alpha_histograms) == [1, 5, 100]
    assert len(result.regime_trace) == result.iterations
    assert len(result.entropy_trace) == result.iterations
    assert result.phi_residual_l1 >= 0
