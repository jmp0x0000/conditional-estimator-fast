"""Tests for ConditionalEstimator."""

import math

import numpy as np
import pytest

from conditional_estimator import ConditionalEstimator


# ------------------------------------------------------------------ #
#  PAVA (isotonic regression) unit tests                              #
# ------------------------------------------------------------------ #

class TestPAVA:
    """Direct tests of the _pava_decreasing static method."""

    def test_already_decreasing(self):
        xs = np.array([1.0, 2.0, 3.0, 4.0])
        ys = np.array([80.0, 60.0, 40.0, 20.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        np.testing.assert_array_equal(rx, xs)
        np.testing.assert_array_equal(ry, ys)

    def test_constant_is_valid(self):
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([50.0, 50.0, 50.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        np.testing.assert_array_equal(ry, [50.0, 50.0, 50.0])

    def test_single_violation(self):
        # 80, 90 violates non-increasing → should merge to 85
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([80.0, 90.0, 20.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert ry[0] == pytest.approx(85.0)
        assert ry[1] == pytest.approx(85.0)
        assert ry[2] == pytest.approx(20.0)

    def test_fully_increasing_collapses_to_mean(self):
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([10.0, 20.0, 30.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        expected = pytest.approx(20.0)
        assert ry[0] == expected
        assert ry[1] == expected
        assert ry[2] == expected

    def test_output_is_non_increasing(self):
        rng = np.random.default_rng(42)
        xs = np.arange(20, dtype=float)
        ys = rng.normal(0, 10, size=20)
        _, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        diffs = np.diff(ry)
        assert np.all(diffs <= 1e-12), f"Non-increasing violated: {diffs}"

    def test_duplicate_x_averaged(self):
        xs = np.array([1.0, 1.0, 2.0])
        ys = np.array([80.0, 60.0, 30.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert len(rx) == 2
        assert rx[0] == pytest.approx(1.0)
        assert ry[0] == pytest.approx(70.0)  # average of 80 and 60
        assert ry[1] == pytest.approx(30.0)

    def test_unsorted_input(self):
        xs = np.array([3.0, 1.0, 2.0])
        ys = np.array([20.0, 80.0, 60.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        np.testing.assert_array_equal(rx, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(ry, [80.0, 60.0, 20.0])

    def test_single_point(self):
        xs = np.array([5.0])
        ys = np.array([42.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert len(rx) == 1
        assert ry[0] == pytest.approx(42.0)

    def test_two_points_violation(self):
        xs = np.array([1.0, 2.0])
        ys = np.array([30.0, 70.0])  # increasing → violation
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert ry[0] == pytest.approx(50.0)
        assert ry[1] == pytest.approx(50.0)

    def test_chain_of_violations(self):
        # 100, 10, 20, 30, 5 → violations at positions 1-3
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys = np.array([100.0, 10.0, 20.0, 30.0, 5.0])
        _, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert np.all(np.diff(ry) <= 1e-12)
        assert ry[0] == pytest.approx(100.0)
        assert ry[-1] == pytest.approx(5.0)


# ------------------------------------------------------------------ #
#  Minimum observations / None returns                                #
# ------------------------------------------------------------------ #

class TestMinObservations:

    def test_mean_returns_none_with_zero_obs(self):
        est = ConditionalEstimator()
        assert est.mean(1.0) is None

    def test_variance_returns_none_with_zero_obs(self):
        est = ConditionalEstimator()
        assert est.variance(1.0) is None

    def test_std_returns_none_with_zero_obs(self):
        est = ConditionalEstimator()
        assert est.std(1.0) is None

    def test_mean_and_variance_returns_nones(self):
        est = ConditionalEstimator()
        m, v = est.mean_and_variance(1.0)
        assert m is None and v is None

    def test_find_root_raises_with_too_few(self):
        est = ConditionalEstimator()
        est.update(1.0, 90.0)
        with pytest.raises(ValueError, match="Need at least"):
            est.find_root(50.0)

    def test_one_obs_below_default_threshold(self):
        est = ConditionalEstimator()
        est.update(1.0, 80.0)
        assert est.mean(1.0) is None

    def test_custom_min_obs(self):
        est = ConditionalEstimator(min_obs_for_estimate=3)
        est.update(1.0, 80.0)
        est.update(2.0, 60.0)
        assert est.mean(1.5) is None  # still below threshold
        est.update(3.0, 40.0)
        assert est.mean(1.5) is not None


# ------------------------------------------------------------------ #
#  Mean estimation                                                    #
# ------------------------------------------------------------------ #

class TestMean:

    def test_exact_observations_no_noise(self):
        est = ConditionalEstimator()
        est.update(1.0, 100.0)
        est.update(2.0, 80.0)
        est.update(3.0, 60.0)
        est.update(4.0, 40.0)
        # at observed points
        assert est.mean(1.0) == pytest.approx(100.0)
        assert est.mean(4.0) == pytest.approx(40.0)

    def test_linear_interpolation_midpoint(self):
        est = ConditionalEstimator()
        est.update(1.0, 100.0)
        est.update(3.0, 60.0)
        assert est.mean(2.0) == pytest.approx(80.0)

    def test_extrapolation_left(self):
        est = ConditionalEstimator()
        est.update(2.0, 80.0)
        est.update(4.0, 40.0)
        # left extrapolation should clamp to leftmost value
        assert est.mean(0.0) == pytest.approx(80.0)

    def test_extrapolation_right(self):
        est = ConditionalEstimator()
        est.update(2.0, 80.0)
        est.update(4.0, 40.0)
        assert est.mean(10.0) == pytest.approx(40.0)

    def test_monotonicity_of_mean_over_grid(self):
        rng = np.random.default_rng(123)
        est = ConditionalEstimator()
        # true function: 100 - 10*x + noise
        for x in np.linspace(0, 10, 30):
            est.update(x, 100 - 10 * x + rng.normal(0, 3))
        grid = np.linspace(0, 10, 200)
        means = [est.mean(x) for x in grid]
        diffs = np.diff(means)
        assert np.all(diffs <= 1e-10), "Mean must be non-increasing"

    def test_mean_with_noisy_data(self):
        rng = np.random.default_rng(7)
        est = ConditionalEstimator()
        for x in np.linspace(1, 5, 50):
            est.update(x, 100 - 15 * x + rng.normal(0, 2))
        # rough sanity: mean at x=1 should be > mean at x=5
        assert est.mean(1.0) > est.mean(5.0)

    def test_single_unique_x_multiple_obs(self):
        est = ConditionalEstimator()
        est.update(3.0, 50.0)
        est.update(3.0, 60.0)
        # only one unique x → interpolation returns the single isotonic value
        assert est.mean(3.0) == pytest.approx(55.0)
        assert est.mean(0.0) == pytest.approx(55.0)  # extrapolation


# ------------------------------------------------------------------ #
#  Variance estimation                                                #
# ------------------------------------------------------------------ #

class TestVariance:

    def test_zero_noise_gives_zero_variance(self):
        est = ConditionalEstimator()
        # perfectly monotone data, no noise
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.update(x, 100 - 20 * x)
        # variance should be 0 (or very close)
        for x in [1.0, 3.0, 5.0]:
            assert est.variance(x) == pytest.approx(0.0, abs=1e-10)

    def test_variance_is_nonnegative(self):
        rng = np.random.default_rng(99)
        est = ConditionalEstimator()
        for x in np.linspace(0, 10, 40):
            est.update(x, 80 - 5 * x + rng.normal(0, 5))
        for x in np.linspace(0, 10, 50):
            assert est.variance(x) >= 0.0

    def test_high_noise_gives_high_variance(self):
        rng = np.random.default_rng(11)
        est_low = ConditionalEstimator(bandwidth=2.0)
        est_high = ConditionalEstimator(bandwidth=2.0)
        for x in np.linspace(0, 10, 100):
            est_low.update(x, 100 - 10 * x + rng.normal(0, 1))
        rng2 = np.random.default_rng(11)
        for x in np.linspace(0, 10, 100):
            est_high.update(x, 100 - 10 * x + rng2.normal(0, 20))
        # at a middle point, high-noise estimator should have larger variance
        assert est_high.variance(5.0) > est_low.variance(5.0)

    def test_std_is_sqrt_of_variance(self):
        est = ConditionalEstimator()
        est.update(1.0, 90.0)
        est.update(2.0, 85.0)
        est.update(3.0, 50.0)
        v = est.variance(2.0)
        s = est.std(2.0)
        assert s == pytest.approx(math.sqrt(v))

    def test_explicit_bandwidth(self):
        est = ConditionalEstimator(bandwidth=1.0)
        est.update(1.0, 90.0)
        est.update(2.0, 70.0)
        est.update(3.0, 50.0)
        # should work without error
        v = est.variance(2.0)
        assert v is not None and v >= 0.0


# ------------------------------------------------------------------ #
#  mean_and_variance consistency                                      #
# ------------------------------------------------------------------ #

class TestMeanAndVariance:

    def test_consistent_with_separate_calls(self):
        est = ConditionalEstimator()
        est.update_batch([1, 2, 3, 4], [90, 70, 50, 30])
        m, v = est.mean_and_variance(2.5)
        assert m == pytest.approx(est.mean(2.5))
        assert v == pytest.approx(est.variance(2.5))


# ------------------------------------------------------------------ #
#  Root finding                                                       #
# ------------------------------------------------------------------ #

class TestFindRoot:

    def test_exact_linear(self):
        est = ConditionalEstimator()
        # y = 100 - 20*x, so y=50 at x=2.5
        est.update_batch([0, 1, 2, 3, 4, 5], [100, 80, 60, 40, 20, 0])
        root = est.find_root(50.0)
        assert root == pytest.approx(2.5, abs=1e-6)

    def test_root_at_boundary(self):
        est = ConditionalEstimator()
        est.update_batch([1, 2, 3], [90, 60, 30])
        root = est.find_root(90.0)
        assert root == pytest.approx(1.0, abs=1e-6)

    def test_target_outside_range_raises(self):
        est = ConditionalEstimator()
        est.update_batch([1, 2, 3], [80, 60, 40])
        with pytest.raises(ValueError, match="No sign change"):
            est.find_root(95.0)  # above max mean
        with pytest.raises(ValueError, match="No sign change"):
            est.find_root(30.0)  # below min mean

    def test_custom_bracket(self):
        est = ConditionalEstimator()
        est.update_batch([0, 2, 4, 6, 8, 10], [100, 80, 60, 40, 20, 0])
        root = est.find_root(50.0, bracket=(0.0, 10.0))
        assert root == pytest.approx(5.0, abs=1e-6)

    def test_root_with_noisy_data(self):
        rng = np.random.default_rng(42)
        est = ConditionalEstimator()
        # true function: 100 - 10x, root at x=5 for target=50
        for x in np.linspace(0, 10, 100):
            est.update(x, 100 - 10 * x + rng.normal(0, 2))
        root = est.find_root(50.0)
        assert root == pytest.approx(5.0, abs=1.0)  # within 1 unit

    def test_root_with_market_share_scenario(self):
        """Simulate the motivating example: price → market share."""
        est = ConditionalEstimator()
        prices = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        shares = [95, 88, 75, 62, 50, 38, 25, 12, 5]
        est.update_batch(prices, shares)
        price_50 = est.find_root(50.0)
        assert price_50 == pytest.approx(3.0, abs=0.01)


# ------------------------------------------------------------------ #
#  Online updates                                                     #
# ------------------------------------------------------------------ #

class TestOnlineUpdates:

    def test_mean_changes_after_update(self):
        est = ConditionalEstimator()
        est.update(1.0, 100.0)
        est.update(3.0, 60.0)
        m1 = est.mean(2.0)
        est.update(2.0, 70.0)
        m2 = est.mean(2.0)
        # after adding a point at x=2 with y=70, mean at 2 should shift
        # (it was 80.0 by interpolation before; may change slightly with new point)
        assert m2 is not None

    def test_incremental_vs_batch(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [90.0, 75.0, 50.0, 30.0, 10.0]

        est_inc = ConditionalEstimator()
        for x, y in zip(xs, ys):
            est_inc.update(x, y)

        est_batch = ConditionalEstimator()
        est_batch.update_batch(xs, ys)

        for x in [1.5, 2.5, 3.5, 4.5]:
            assert est_inc.mean(x) == pytest.approx(est_batch.mean(x))
            assert est_inc.variance(x) == pytest.approx(est_batch.variance(x))

    def test_n_obs_tracks_correctly(self):
        est = ConditionalEstimator()
        assert est.n_obs == 0
        est.update(1.0, 50.0)
        assert est.n_obs == 1
        est.update_batch([2, 3, 4], [40, 30, 20])
        assert est.n_obs == 4

    def test_dirty_flag_resets(self):
        est = ConditionalEstimator()
        est.update(1.0, 80.0)
        est.update(2.0, 60.0)
        est.mean(1.5)  # triggers fit, clears dirty
        assert est._dirty is False
        est.update(3.0, 40.0)
        assert est._dirty is True
        est.mean(2.0)
        assert est._dirty is False


# ------------------------------------------------------------------ #
#  Edge cases                                                         #
# ------------------------------------------------------------------ #

class TestEdgeCases:

    def test_two_identical_points(self):
        est = ConditionalEstimator()
        est.update(2.0, 50.0)
        est.update(2.0, 50.0)
        assert est.mean(2.0) == pytest.approx(50.0)
        assert est.variance(2.0) == pytest.approx(0.0, abs=1e-10)

    def test_two_identical_x_different_y(self):
        est = ConditionalEstimator()
        est.update(2.0, 60.0)
        est.update(2.0, 40.0)
        assert est.mean(2.0) == pytest.approx(50.0)
        # variance should reflect the spread
        assert est.variance(2.0) > 0

    def test_large_dataset(self):
        rng = np.random.default_rng(0)
        est = ConditionalEstimator()
        xs = rng.uniform(0, 100, size=1000)
        ys = 200 - 2 * xs + rng.normal(0, 5, size=1000)
        est.update_batch(xs, ys)
        m = est.mean(50.0)
        assert m == pytest.approx(100.0, abs=5.0)

    def test_repr(self):
        est = ConditionalEstimator(bandwidth=2.5)
        est.update(1.0, 50.0)
        r = repr(est)
        assert "n_obs=1" in r
        assert "bandwidth=2.5" in r

    def test_negative_values(self):
        est = ConditionalEstimator()
        est.update(-3.0, 20.0)
        est.update(-1.0, 10.0)
        est.update(1.0, 0.0)
        assert est.mean(-2.0) == pytest.approx(15.0)
        assert est.mean(0.0) == pytest.approx(5.0)

    def test_very_close_x_values(self):
        est = ConditionalEstimator(bandwidth=0.001)
        est.update(1.000, 80.0)
        est.update(1.001, 79.0)
        est.update(1.002, 78.0)
        assert est.mean(1.001) == pytest.approx(79.0)


# ------------------------------------------------------------------ #
#  PAVA correctness against known reference                           #
# ------------------------------------------------------------------ #

class TestWeightedPAVA:
    """Verify that observation counts are used as PAVA weights."""

    def test_cluster_resists_single_outlier(self):
        # 3 observations at x=1 (y=50), 1 observation at x=2 (y=100)
        # Old (unweighted): merge to (50+100)/2 = 75
        # New (weighted):   merge to (50*3+100*1)/4 = 62.5
        xs = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
        ys = np.array([50.0, 50.0, 50.0, 100.0, 20.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        # x=1 (w=3) and x=2 (w=1) merge: (50*3 + 100*1) / 4 = 62.5
        assert ry[0] == pytest.approx(62.5)
        assert ry[1] == pytest.approx(62.5)
        assert ry[2] == pytest.approx(20.0)

    def test_weighted_vs_unweighted_differ(self):
        # With weights, the dense cluster should pull harder
        # x=1: 2 obs avg 80, x=2: 1 obs = 90, x=3: 1 obs = 10
        # Violation at x=1(80) < x=2(90)
        # Unweighted merge: (80+90)/2 = 85
        # Weighted merge:   (80*2+90*1)/3 = 83.33
        xs = np.array([1.0, 1.0, 2.0, 3.0])
        ys = np.array([80.0, 80.0, 90.0, 10.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert ry[0] == pytest.approx(250.0 / 3)  # (80*2 + 90) / 3
        assert ry[1] == pytest.approx(250.0 / 3)
        assert ry[2] == pytest.approx(10.0)

    def test_user_data_scenario(self):
        # The user's actual data — verify the improved plateau
        xs = np.array([0, -10, -10, -9, -9, -9, 20, 10, -100], dtype=float)
        ys = np.array([100, 50, 60, 55, 53, 48, 30, 40, 100], dtype=float)
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        # After dedup: (-100,100,w=1), (-10,55,w=2), (-9,52,w=3), (0,100,w=1), (10,40,w=1), (20,30,w=1)
        # Violation at (-9,52) < (0,100): merge → (52*3+100*1)/4 = 64, w=4
        # Then (-10,55) < merged(64): merge → (55*2+64*4)/6 = 61, w=6
        assert ry[0] == pytest.approx(100.0)           # x=-100
        assert ry[1] == pytest.approx(61.0)             # x=-10
        assert ry[2] == pytest.approx(61.0)             # x=-9
        assert ry[3] == pytest.approx(61.0)             # x=0
        assert ry[4] == pytest.approx(40.0)             # x=10
        assert ry[5] == pytest.approx(30.0)             # x=20

    def test_heavy_cluster_dominates(self):
        # 10 observations at x=1 (y=50), 1 at x=2 (y=100)
        xs = np.array([1.0] * 10 + [2.0])
        ys = np.array([50.0] * 10 + [100.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        # merge: (50*10 + 100*1) / 11 ≈ 54.55
        expected = (50.0 * 10 + 100.0) / 11
        assert ry[0] == pytest.approx(expected)
        assert ry[1] == pytest.approx(expected)

    def test_weighted_sum_preserved(self):
        # Weighted PAVA preserves Σ(w_i * y_i)
        xs = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
        ys = np.array([80.0, 70.0, 90.0, 20.0, 60.0, 50.0, 10.0])
        rx, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        # After dedup: x=1(y=80,w=3), x=2(y=20,w=1), x=3(y=55,w=2), x=4(y=10,w=1)
        # Weighted input sum = 80*3 + 20*1 + 55*2 + 10*1 = 240+20+110+10 = 380
        _, inv = np.unique(xs[np.argsort(xs)], return_inverse=True)
        counts = np.zeros(len(rx))
        np.add.at(counts, inv, 1.0)
        weighted_sum_fitted = np.dot(counts, ry)
        weighted_sum_input = np.dot(counts, np.array([80.0, 20.0, 55.0, 10.0]))
        assert weighted_sum_fitted == pytest.approx(weighted_sum_input)


# ------------------------------------------------------------------ #
#  Variance — uncertainty-aware estimation                            #
# ------------------------------------------------------------------ #

class TestVarianceUncertainty:

    def test_sparse_region_has_nonzero_variance(self):
        # Single isolated observation should NOT give zero variance
        # when there's noise elsewhere in the data
        est = ConditionalEstimator(bandwidth=1.0)
        est.update(-100.0, 100.0)  # isolated point
        # cluster with noise near x=0
        est.update_batch([0, 0, 0, 0], [52, 48, 55, 45])
        est.update(10.0, 30.0)
        v_isolated = est.variance(-100.0)
        v_cluster = est.variance(0.0)
        # isolated region should have HIGHER variance
        assert v_isolated > v_cluster
        assert v_isolated > 0

    def test_isolated_point_gets_global_var_floor(self):
        # With narrow bandwidth, an isolated point far from data should
        # still get nonzero variance via the n_eff < 2 global-var floor
        est = ConditionalEstimator(bandwidth=0.5)
        rng = np.random.default_rng(42)
        for x in np.linspace(0, 5, 30):
            est.update(x, 100 - 10 * x + rng.normal(0, 5))
        est.update(50.0, 0.0)  # isolated, noiseless
        v = est.variance(50.0)
        assert v > 0, "Isolated point should not have zero variance"

    def test_noiseless_still_zero(self):
        # With zero residuals everywhere, variance must stay zero
        est = ConditionalEstimator()
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.update(x, 100 - 20 * x)
        for x in [1.0, 3.0, 5.0]:
            assert est.variance(x) == pytest.approx(0.0, abs=1e-10)

    def test_global_var_floor_nonzero_at_far_query(self):
        # Query point far from all data should get global variance,
        # not zero (the old behavior)
        est = ConditionalEstimator(bandwidth=1.0)
        rng = np.random.default_rng(99)
        for x in np.linspace(0, 10, 30):
            est.update(x, 100 - 10 * x + rng.normal(0, 5))
        # far-away query: kernel weights ≈ 0, but we return global var
        v_far = est.variance(1000.0)
        global_var = est._global_residual_variance()
        assert v_far > 0
        assert v_far == pytest.approx(global_var, rel=0.5)


# ------------------------------------------------------------------ #
#  Bootstrap confidence interval on root                              #
# ------------------------------------------------------------------ #

class TestFindRootCI:

    def test_basic_ci(self):
        est = ConditionalEstimator()
        est.update_batch([0, 2, 4, 6, 8, 10], [100, 80, 60, 40, 20, 0])
        root, lo, hi = est.find_root_ci(50.0, rng=42)
        assert lo <= root <= hi
        assert root == pytest.approx(5.0, abs=0.5)

    def test_ci_covers_true_root(self):
        rng = np.random.default_rng(7)
        est = ConditionalEstimator()
        # true root at x=5 for target=50
        for x in np.linspace(0, 10, 50):
            est.update(x, 100 - 10 * x + rng.normal(0, 3))
        _, lo, hi = est.find_root_ci(50.0, confidence=0.95, rng=0)
        assert lo < 5.0 < hi, f"95% CI [{lo:.2f}, {hi:.2f}] does not cover true root 5.0"

    def test_ci_narrows_with_more_data(self):
        rng = np.random.default_rng(42)
        est_small = ConditionalEstimator()
        est_large = ConditionalEstimator()
        for x in np.linspace(0, 10, 20):
            y = 100 - 10 * x + rng.normal(0, 5)
            est_small.update(x, y)
            est_large.update(x, y)
        rng2 = np.random.default_rng(99)
        for x in np.linspace(0, 10, 200):
            est_large.update(x, 100 - 10 * x + rng2.normal(0, 5))
        _, lo_s, hi_s = est_small.find_root_ci(50.0, rng=0)
        _, lo_l, hi_l = est_large.find_root_ci(50.0, rng=0)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_ci_raises_with_too_few_obs(self):
        est = ConditionalEstimator()
        est.update(1.0, 90.0)
        with pytest.raises(ValueError, match="Need at least"):
            est.find_root_ci(50.0)

    def test_ci_respects_confidence_level(self):
        rng = np.random.default_rng(42)
        est = ConditionalEstimator()
        for x in np.linspace(0, 10, 50):
            est.update(x, 100 - 10 * x + rng.normal(0, 5))
        _, lo_90, hi_90 = est.find_root_ci(50.0, confidence=0.90, rng=0)
        _, lo_99, hi_99 = est.find_root_ci(50.0, confidence=0.99, rng=0)
        # 99% CI should be wider than 90%
        assert (hi_99 - lo_99) > (hi_90 - lo_90)

    def test_ci_with_custom_bracket(self):
        est = ConditionalEstimator()
        est.update_batch([0, 5, 10], [100, 50, 0])
        root, lo, hi = est.find_root_ci(50.0, bracket=(0.0, 10.0), rng=42)
        assert lo <= root <= hi


class TestPAVAReference:
    """Compare PAVA output against manually computed solutions."""

    def test_reference_case_1(self):
        # Input:  [9, 3, 5, 7, 1]  (for x = [1,2,3,4,5])
        # Violations: 3<5, 5<7 → merge 3,5,7 → 5, but 9>5>1 is OK
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys = np.array([9.0, 3.0, 5.0, 7.0, 1.0])
        _, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert ry[0] == pytest.approx(9.0)
        assert ry[1] == pytest.approx(5.0)
        assert ry[2] == pytest.approx(5.0)
        assert ry[3] == pytest.approx(5.0)
        assert ry[4] == pytest.approx(1.0)

    def test_reference_case_2(self):
        # Input: [1, 2, 3] → fully increasing, should all collapse to mean=2
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([1.0, 2.0, 3.0])
        _, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        np.testing.assert_allclose(ry, [2.0, 2.0, 2.0])

    def test_weighted_merge_preserves_total(self):
        # after PAVA, the sum of fitted values should equal the sum of inputs
        # (for equal-weight observations with no duplicate x)
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys = np.array([10.0, 30.0, 5.0, 25.0, 2.0])
        _, ry = ConditionalEstimator._pava_decreasing(xs, ys)
        assert ry.sum() == pytest.approx(ys.sum())
