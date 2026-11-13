"""Tests for the correlated batch combination module."""

import numpy as np
import pytest

from uncertainties import UFloat

from uncertainties.batch_combine import (
    BatchResult,
    ErrorSource,
    build_measurements,
    combine_batches,
    covariance_from_sources,
    optimal_simplex_weights,
)


SIMPLEX_TOL = 1e-8


def assert_simplex_weights(weights):
    assert np.all(weights >= -1e-12)
    assert weights.sum() == pytest.approx(1.0)


def assert_variance_optimal(cov, weights, variance):
    """No random simplex point may beat the claimed optimal variance."""
    rng = np.random.default_rng(0)
    n = cov.shape[0]
    for _ in range(20000):
        sample = rng.dirichlet(np.ones(n))
        assert sample @ cov @ sample >= variance - 1e-10


def test_single_batch_gets_unit_weight():
    cov = np.array([[0.25]])
    weights, variance = optimal_simplex_weights(cov)
    assert weights == pytest.approx([1.0])
    assert variance == pytest.approx(0.25)


def test_independent_equal_batches_share_weight_evenly():
    cov = np.diag([0.1, 0.1, 0.1])
    weights, variance = optimal_simplex_weights(cov)
    assert weights == pytest.approx(np.full(3, 1.0 / 3.0))
    assert variance == pytest.approx(0.1 / 3.0)
    assert_variance_optimal(cov, weights, variance)


def test_independent_batches_recover_inverse_variance_weights():
    variances = np.array([0.5, 1.0, 2.0])
    cov = np.diag(variances)
    weights, variance = optimal_simplex_weights(cov)
    expected = (1.0 / variances) / (1.0 / variances).sum()
    assert weights == pytest.approx(expected)
    assert variance == pytest.approx(1.0 / (1.0 / variances).sum())
    assert_simplex_weights(weights)
    assert_variance_optimal(cov, weights, variance)


def test_negative_unconstrained_solution_is_clipped_to_zero():
    # Batch 2 is so noisy that unconstrained inverse variance would still
    # keep it positive; instead engineer a case requiring clipping through
    # correlation: a strongly anti-correlated pair plus one solid batch.
    cov = np.array(
        [
            [1.0, -0.9, 0.0],
            [-0.9, 1.0, 0.0],
            [0.0, 0.0, 0.1],
        ]
    )
    weights, variance = optimal_simplex_weights(cov)
    assert_simplex_weights(weights)
    assert np.all(weights >= 0.0)
    assert_variance_optimal(cov, weights, variance)


def test_perfectly_correlated_duplicates_do_not_add_precision():
    # One genuine measurement and two fully redundant copies of it.
    cov = np.array(
        [
            [0.3, 0.3, 0.0],
            [0.3, 0.3, 0.0],
            [0.0, 0.0, 0.6],
        ]
    )
    weights, variance = optimal_simplex_weights(cov)
    assert_simplex_weights(weights)
    # The two copies must split the weight a single copy would receive.
    single_cov = np.array([[0.3, 0.0], [0.0, 0.6]])
    single_weights, single_variance = optimal_simplex_weights(single_cov)
    assert weights[0] + weights[1] == pytest.approx(single_weights[0])
    assert weights[0] == pytest.approx(weights[1])
    assert variance == pytest.approx(single_variance)
    assert_variance_optimal(cov, weights, variance)


def test_tie_break_prefers_minimum_squared_norm():
    # Three identical rows: every convex combination of the three copies is
    # variance-optimal, so the minimum squared-norm choice is the uniform one.
    cov = np.full((3, 3), 0.4)
    weights, variance = optimal_simplex_weights(cov)
    assert weights == pytest.approx(np.full(3, 1.0 / 3.0))
    assert variance == pytest.approx(0.4)


def test_shared_calibration_covariance_is_built_from_source_identity():
    batches = [
        BatchResult("B1", 10.0, 0.2),
        BatchResult("B2", 11.0, 0.3),
        BatchResult("B3", 12.0, 0.4),
    ]
    sources = [
        ErrorSource("CAL", 0.5, ("B1", "B2")),
        ErrorSource("SMP", 0.1, ("B2", "B3")),
    ]
    cov = covariance_from_sources(batches, sources)
    expected = np.array(
        [
            [0.2**2 + 0.5**2, 0.5**2, 0.0],
            [0.5**2, 0.3**2 + 0.5**2 + 0.1**2, 0.1**2],
            [0.0, 0.1**2, 0.4**2 + 0.1**2],
        ]
    )
    assert cov == pytest.approx(expected)


def test_duplicate_submission_is_perfectly_correlated():
    # A resubmitted sample brings no new private information: all of its
    # variance comes from sources shared with the first submission.
    batches = [
        BatchResult("B4", 10.1, 0.0),
        BatchResult("B5", 10.1, 0.0),
    ]
    sources = [
        ErrorSource("CAL-B", 0.4, ("B4", "B5")),
        ErrorSource("SMP-X", 0.15, ("B4", "B5")),
    ]
    measurements = build_measurements(batches, sources)
    assert measurements[0].std_dev == pytest.approx(
        np.sqrt(0.4**2 + 0.15**2)
    )
    covariance = covariance_from_sources(batches, sources)
    correlation = covariance[0, 1] / np.sqrt(
        covariance[0, 0] * covariance[1, 1]
    )
    assert correlation == pytest.approx(1.0)


def test_combine_batches_end_to_end():
    batches = [
        BatchResult("B1", 10.0, 0.2),
        BatchResult("B2", 10.2, 0.1),
        BatchResult("B3", 10.1, 0.1),
    ]
    sources = [ErrorSource("CAL", 0.3, ("B2", "B3"))]
    result = combine_batches(batches, sources)

    assert result["batch_ids"] == ("B1", "B2", "B3")
    assert_simplex_weights(result["weights"])
    expected_value = float(result["weights"] @ np.array([10.0, 10.2, 10.1]))
    assert result["combined_value"] == pytest.approx(expected_value)
    assert result["combined_std_dev"] == pytest.approx(
        np.sqrt(result["combined_variance"])
    )
    assert result["combined_variance"] == pytest.approx(
        result["weights"] @ result["covariance"] @ result["weights"]
    )
    assert_variance_optimal(
        result["covariance"], result["weights"], result["combined_variance"]
    )


def test_random_psd_matrices_always_satisfy_optimality():
    rng = np.random.default_rng(42)
    for n in range(2, 7):
        for _ in range(30):
            random_matrix = rng.normal(size=(n, n))
            cov = random_matrix @ random_matrix.T / n
            weights, variance = optimal_simplex_weights(cov)
            assert_simplex_weights(weights)
            # KKT check: gradients on positive-weight indices coincide and
            # dominate every zero-weight index.
            gradient = cov @ weights
            positive = weights > 1e-10
            lagrange = gradient[positive][0]
            assert gradient[positive] == pytest.approx(
                lagrange, abs=1e-8
            )
            assert np.all(gradient[~positive] >= lagrange - 1e-8)
            assert_variance_optimal(cov, weights, variance)


def test_rank_deficient_covariance_does_not_crash():
    # Two perfectly correlated rows plus an independent one.
    cov = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    weights, variance = optimal_simplex_weights(cov)
    assert_simplex_weights(weights)
    assert weights[0] == pytest.approx(weights[1])
    assert_variance_optimal(cov, weights, variance)


def test_large_batch_count_iterative_path_is_optimal():
    # Past the exact-enumeration threshold the ridge Frank-Wolfe path must
    # still satisfy the simplex and KKT optimality conditions.
    rng = np.random.default_rng(2024)
    n = 20
    factor = rng.normal(size=(n, n))
    cov = factor @ factor.T
    weights, variance = optimal_simplex_weights(cov)

    assert_simplex_weights(weights)
    assert variance >= 0.0
    gradient = cov @ weights
    positive = weights > 1e-9
    lagrange = gradient[positive][0]
    assert gradient[positive] == pytest.approx(lagrange, abs=1e-6)
    assert np.all(gradient[~positive] >= lagrange - 1e-6)
    assert_variance_optimal(cov, weights, variance)


def test_large_duplicate_submissions_split_their_weight_equally():
    # 18 batches where the first four share an identical covariance row.
    rng = np.random.default_rng(11)
    n = 18
    factor = rng.normal(size=(n, 6))
    for index in range(1, 4):
        factor[index] = factor[0]
    cov = factor @ factor.T

    weights, variance = optimal_simplex_weights(cov)
    assert_simplex_weights(weights)
    assert weights[:4] == pytest.approx(weights[0])
    assert_variance_optimal(cov, weights, variance)


def test_input_validation():
    with pytest.raises(ValueError):
        ErrorSource("CAL", -1.0, ("B1",))
    with pytest.raises(ValueError):
        BatchResult("B1", 1.0, -0.1)
    with pytest.raises(ValueError):
        optimal_simplex_weights(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        optimal_simplex_weights(np.zeros((0, 0)))
    duplicate_batches = [
        BatchResult("B1", 1.0),
        BatchResult("B1", 2.0),
    ]
    with pytest.raises(ValueError):
        build_measurements(duplicate_batches, [])


def test_measurements_are_tagged_ufloats():
    batches = [BatchResult("B1", 1.0, 0.1)]
    sources = [ErrorSource("CAL", 0.2, ("B1",))]
    measurement = build_measurements(batches, sources)[0]
    tags = {variable.tag for variable in measurement.error_components()}
    assert "private:B1" in tags
    assert "CAL" in tags
    assert isinstance(measurement, UFloat)
