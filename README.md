# conditional-estimator

Monotone conditional mean and variance estimator with root finding. Designed for online settings where observations arrive dynamically — e.g. estimating at which price market share reaches 50%.

## What it does

Given scattered observations of a **(parameter, target)** pair where the target is monotonically decreasing in the parameter:

1. **Estimates the conditional mean** E[Y|X=x] using weighted isotonic regression (PAVA), enforcing monotonicity even when individual observations are noisy or out of order.
2. **Estimates the conditional variance** Var[Y|X=x] using kernel-weighted local residuals, with uncertainty-aware inflation at sparse regions.
3. **Finds roots** — the parameter value where the mean hits a target level — with optional bootstrap confidence intervals.

All three update online as new observations arrive.

## Installation

```bash
pip install numpy scipy
```

No additional dependencies. Copy `conditional_estimator.py` into your project, or install from source:

```bash
pip install .
```

## Quick start

```python
from conditional_estimator import ConditionalEstimator

est = ConditionalEstimator()

# Observations arrive over time: (price, market_share%)
est.update(1.0, 95)
est.update(2.0, 75)
est.update(3.0, 50)
est.update(4.0, 25)
est.update(5.0, 5)

# Query the estimated mean and std at any point
est.mean(2.5)       # ~62.5
est.std(2.5)        # local noise estimate

# Find the price where market share = 50%
est.find_root(50.0)  # 3.0

# With a 95% bootstrap confidence interval
root, ci_lo, ci_hi = est.find_root_ci(50.0)
```

## API

### `ConditionalEstimator(bandwidth=None, min_obs_for_estimate=2)`

| Parameter | Description |
|-----------|-------------|
| `bandwidth` | Kernel bandwidth for local variance estimation. `None` = auto (0.5 * median spacing). |
| `min_obs_for_estimate` | Minimum observations before `mean`/`variance` return values (otherwise `None`). |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `update(x, y)` | `None` | Add a single observation. |
| `update_batch(xs, ys)` | `None` | Add multiple observations. |
| `mean(x)` | `float \| None` | Conditional mean at x (isotonic + linear interpolation). |
| `variance(x)` | `float \| None` | Conditional variance at x (kernel-weighted, uncertainty-aware). |
| `std(x)` | `float \| None` | Conditional standard deviation at x. |
| `mean_and_variance(x)` | `(float, float) \| (None, None)` | Both in a single call (avoids double refit). |
| `find_root(target)` | `float` | Parameter value where mean equals target (Brent's method). |
| `find_root_ci(target, confidence=0.95)` | `(root, ci_lo, ci_hi)` | Root with bootstrap confidence interval. |

### Properties

| Property | Description |
|----------|-------------|
| `n_obs` | Number of observations added so far. |

## How it works

### Weighted isotonic regression (PAVA)

Observations are fitted using the Pool Adjacent Violators Algorithm with **observation-count weighting**. When multiple observations share the same x-value, they're averaged and their count is carried as weight into the PAVA merge step. This means a cluster of 5 observations at x=-9 averaging 52 will resist being pulled up by a single outlier at x=0 with value 100 — the merged plateau reflects the density of evidence, not just the number of unique x-values.

Between fitted points, the mean is linearly interpolated. Beyond the observed range, it extrapolates flat (constant at the boundary value).

### Variance estimation

Local variance is estimated using Gaussian kernel-weighted squared residuals from the isotonic fit. The total variance includes an estimation uncertainty term:

```
var_total = var_noise * (1 + 1/n_eff)
```

where `n_eff` is the kernel-effective sample size. This prevents false confidence at sparse regions — an isolated observation won't report zero variance just because its residual happens to be zero.

When `n_eff < 2`, the estimator falls back to the global residual variance as a floor.

### Root finding

`find_root` uses `scipy.optimize.brentq` on the interpolated mean function. `find_root_ci` adds a nonparametric bootstrap: it resamples observations with replacement, refits, and finds the root on each resample to build a percentile confidence interval.

## Example with real-world-like data

```python
import numpy as np
from conditional_estimator import ConditionalEstimator

est = ConditionalEstimator()

# Sparse, noisy observations
est.update_batch(
    [0, -10, -10, -9, -9, -9, 20, 10, -100],
    [100, 50, 60, 55, 53, 48, 30, 40, 100],
)

# Isotonic fit respects observation density
print(est.mean(-9))   # ~61 (not 69 — the cluster's weight matters)
print(est.mean(10))   # 40

# Root with confidence interval
root, lo, hi = est.find_root_ci(50.0, rng=42)
print(f"50% at x={root:.1f}, 95% CI [{lo:.1f}, {hi:.1f}]")
```

## Running tests

```bash
pip install pytest
pytest test_conditional_estimator.py -v
```

## License

MIT
