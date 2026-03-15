"""Conditional mean and variance estimator for a monotonically decreasing function.

Maintains online observations of (parameter, target) pairs and provides
isotonic-regression-based estimates of the conditional mean and local
estimates of the conditional variance.  Designed for root-finding use cases
(e.g. "at which price does market share equal 50%?").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


@dataclass
class ConditionalEstimator:
    """Estimate E[Y|X=x] and Var[Y|X=x] under a monotone-decreasing constraint.

    Parameters
    ----------
    bandwidth : float or None
        Kernel bandwidth for local variance estimation.  If *None* (default),
        bandwidth is set automatically to ``0.5 * median_spacing`` of the
        observed x values (recomputed on every query once there are ≥ 3 points).
    min_obs_for_estimate : int
        Minimum number of observations required before ``mean`` / ``variance``
        return finite values.  Below this threshold they return *None*.
    """

    bandwidth: Optional[float] = None
    min_obs_for_estimate: int = 2

    # -- internal state --
    _xs: list[float] = field(default_factory=list, repr=False)
    _ys: list[float] = field(default_factory=list, repr=False)
    _dirty: bool = field(default=True, repr=False)
    _iso_xs: np.ndarray | None = field(default=None, repr=False)
    _iso_ys: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def update(self, x: float, y: float) -> None:
        """Add a new (parameter, target) observation."""
        self._xs.append(float(x))
        self._ys.append(float(y))
        self._dirty = True

    def update_batch(self, xs, ys) -> None:
        """Add several observations at once."""
        for x, y in zip(xs, ys):
            self.update(x, y)

    @property
    def n_obs(self) -> int:
        return len(self._xs)

    def mean(self, x: float) -> Optional[float]:
        """Return the estimated conditional mean E[Y|X=x], or *None*."""
        if self.n_obs < self.min_obs_for_estimate:
            return None
        self._ensure_fit()
        return float(self._interpolate_mean(x))

    def variance(self, x: float) -> Optional[float]:
        """Return the estimated conditional variance Var[Y|X=x], or *None*."""
        if self.n_obs < self.min_obs_for_estimate:
            return None
        self._ensure_fit()
        return float(self._local_variance(x))

    def std(self, x: float) -> Optional[float]:
        """Return the estimated conditional standard deviation, or *None*."""
        v = self.variance(x)
        return math.sqrt(v) if v is not None else None

    def mean_and_variance(self, x: float) -> tuple[Optional[float], Optional[float]]:
        """Return (mean, variance) in a single call (avoids double refit)."""
        if self.n_obs < self.min_obs_for_estimate:
            return None, None
        self._ensure_fit()
        return float(self._interpolate_mean(x)), float(self._local_variance(x))

    def find_root(
        self,
        target: float,
        bracket: tuple[float, float] | None = None,
        **brentq_kwargs,
    ) -> float:
        """Find *x* such that E[Y|X=x] ≈ target.

        Parameters
        ----------
        target : float
            The desired conditional-mean level (e.g. 50 for 50 % market share).
        bracket : (lo, hi) or None
            Search interval.  If *None*, the range of observed x values is used.
        **brentq_kwargs
            Forwarded to :func:`scipy.optimize.brentq` (e.g. ``xtol``, ``maxiter``).

        Returns
        -------
        float
            The estimated parameter value.

        Raises
        ------
        ValueError
            If there are too few observations or the target is outside the
            range of fitted mean values (no sign change).
        """
        if self.n_obs < self.min_obs_for_estimate:
            raise ValueError(
                f"Need at least {self.min_obs_for_estimate} observations, "
                f"have {self.n_obs}."
            )
        self._ensure_fit()

        if bracket is None:
            lo, hi = float(self._iso_xs[0]), float(self._iso_xs[-1])
        else:
            lo, hi = bracket

        def f(x: float) -> float:
            return self._interpolate_mean(x) - target

        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            raise ValueError(
                f"No sign change on [{lo}, {hi}]: "
                f"mean({lo})={f_lo + target:.4f}, mean({hi})={f_hi + target:.4f}.  "
                f"Target {target} may be outside the estimable range."
            )

        brentq_kwargs.setdefault("xtol", 1e-8)
        return float(brentq(f, lo, hi, **brentq_kwargs))

    def find_root_ci(
        self,
        target: float,
        confidence: float = 0.95,
        n_bootstrap: int = 2000,
        bracket: tuple[float, float] | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> tuple[float, float, float]:
        """Find root with a bootstrap confidence interval.

        Parameters
        ----------
        target : float
            The desired conditional-mean level.
        confidence : float
            Confidence level (e.g. 0.95 for 95 %).
        n_bootstrap : int
            Number of bootstrap resamples.
        bracket : (lo, hi) or None
            Search interval; defaults to the observed x range.
        rng : Generator, int, or None
            Random state for reproducibility.

        Returns
        -------
        (root, ci_lo, ci_hi) : tuple[float, float, float]
            Point estimate and confidence interval bounds.
        """
        if self.n_obs < self.min_obs_for_estimate:
            raise ValueError(
                f"Need at least {self.min_obs_for_estimate} observations, "
                f"have {self.n_obs}."
            )

        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()

        # point estimate
        root = self.find_root(target, bracket=bracket)

        xs = np.asarray(self._xs)
        ys = np.asarray(self._ys)
        n = len(xs)

        if bracket is None:
            self._ensure_fit()
            br = (float(self._iso_xs[0]), float(self._iso_xs[-1]))
        else:
            br = bracket

        roots = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bx, by = xs[idx], ys[idx]

            tmp = ConditionalEstimator(
                bandwidth=self.bandwidth,
                min_obs_for_estimate=self.min_obs_for_estimate,
            )
            tmp.update_batch(bx, by)

            try:
                r = tmp.find_root(target, bracket=br)
                roots.append(r)
            except ValueError:
                # target outside range for this resample — skip
                continue

        if len(roots) < 10:
            raise ValueError(
                f"Only {len(roots)}/{n_bootstrap} bootstrap samples produced "
                f"a valid root.  Target {target} may be near the edge of the "
                f"estimable range."
            )

        alpha = 1.0 - confidence
        ci_lo = float(np.percentile(roots, 100 * alpha / 2))
        ci_hi = float(np.percentile(roots, 100 * (1 - alpha / 2)))

        return root, ci_lo, ci_hi

    # ------------------------------------------------------------------ #
    #  Isotonic regression (Pool Adjacent Violators — decreasing)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pava_decreasing(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (sorted_xs, isotonic_ys) enforcing a non-increasing fit.

        When several observations share the same *x* they are averaged first.
        """
        order = np.argsort(xs)
        xs_sorted = xs[order].copy()
        ys_sorted = ys[order].copy()

        # average duplicates
        ux, inv = np.unique(xs_sorted, return_inverse=True)
        uy = np.zeros_like(ux, dtype=float)
        counts = np.zeros_like(ux, dtype=float)
        np.add.at(uy, inv, ys_sorted)
        np.add.at(counts, inv, 1.0)
        uy /= counts
        xs_sorted = ux
        ys_sorted = uy

        # PAVA for non-increasing sequence (observation-count weighted)
        n = len(ys_sorted)
        blocks_val = list(ys_sorted)
        blocks_w = list(counts)
        blocks_idx = list(range(n))  # start index of each block

        i = 0
        while i < len(blocks_val) - 1:
            if blocks_val[i] < blocks_val[i + 1]:  # violation of non-increasing
                # merge i and i+1
                w = blocks_w[i] + blocks_w[i + 1]
                v = (blocks_val[i] * blocks_w[i] + blocks_val[i + 1] * blocks_w[i + 1]) / w
                blocks_val[i] = v
                blocks_w[i] = w
                del blocks_val[i + 1]
                del blocks_w[i + 1]
                del blocks_idx[i + 1]
                # step back to check previous
                if i > 0:
                    i -= 1
            else:
                i += 1

        # expand blocks back to per-point values
        iso_ys = np.empty(n)
        for k in range(len(blocks_idx)):
            start = blocks_idx[k]
            end = blocks_idx[k + 1] if k + 1 < len(blocks_idx) else n
            iso_ys[start:end] = blocks_val[k]

        return xs_sorted, iso_ys

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_fit(self) -> None:
        if not self._dirty:
            return
        xs = np.asarray(self._xs)
        ys = np.asarray(self._ys)
        self._iso_xs, self._iso_ys = self._pava_decreasing(xs, ys)
        self._dirty = False

    def _interpolate_mean(self, x: float) -> float:
        """Linearly interpolate the isotonic fit, constant-extrapolating."""
        if len(self._iso_xs) == 1:
            return float(self._iso_ys[0])
        f = interp1d(
            self._iso_xs,
            self._iso_ys,
            kind="linear",
            fill_value=(self._iso_ys[0], self._iso_ys[-1]),
            bounds_error=False,
        )
        return float(f(x))

    def _effective_bandwidth(self) -> float:
        if self.bandwidth is not None:
            return self.bandwidth
        if len(self._iso_xs) < 3:
            # fallback: half the data range
            return (self._iso_xs[-1] - self._iso_xs[0]) / 2.0 or 1.0
        diffs = np.diff(self._iso_xs)
        return float(0.5 * np.median(diffs)) or 1.0

    def _global_residual_variance(self) -> float:
        """Variance of all residuals (y - fitted) across observations."""
        xs = np.asarray(self._xs)
        ys = np.asarray(self._ys)
        fitted = np.array([self._interpolate_mean(xi) for xi in xs])
        residuals = ys - fitted
        if len(residuals) > 1:
            return float(np.var(residuals, ddof=1))
        return 0.0

    def _local_variance(self, x: float) -> float:
        """Kernel-weighted local variance around *x*.

        Uses a Gaussian kernel with the effective bandwidth.  The variance
        is computed against the *isotonic mean* (not raw y), so it reflects
        the noise around the monotone trend.

        The total variance has two components:
          1. Local noise variance — scatter of nearby observations around the fit.
          2. Estimation uncertainty — inversely proportional to the effective
             number of nearby observations (n_eff).  This prevents false
             confidence at sparse regions.
        """
        xs = np.asarray(self._xs)
        ys = np.asarray(self._ys)
        h = self._effective_bandwidth()

        # Gaussian kernel weights
        weights = np.exp(-0.5 * ((xs - x) / h) ** 2)
        w_sum = weights.sum()

        global_var = self._global_residual_variance()

        if w_sum < 1e-12:
            # query far from data — return global variance
            return global_var if global_var > 0 else 0.0

        # effective sample size: (sum w)^2 / sum(w^2)
        n_eff = w_sum ** 2 / np.dot(weights, weights)

        # local noise: weighted variance of residuals
        fitted = np.array([self._interpolate_mean(xi) for xi in xs])
        residuals = ys - fitted
        wm = np.dot(weights, residuals) / w_sum
        var_noise = np.dot(weights, (residuals - wm) ** 2) / w_sum

        if n_eff < 2.0:
            # too few effective observations locally — use global variance
            var_noise = max(var_noise, global_var)

        # estimation uncertainty: noise / n_eff
        var_total = var_noise * (1.0 + 1.0 / n_eff)

        return float(max(var_total, 0.0))

    # ------------------------------------------------------------------ #
    #  Convenience                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"ConditionalEstimator(n_obs={self.n_obs}, "
            f"bandwidth={self.bandwidth})"
        )
