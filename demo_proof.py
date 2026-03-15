"""Demonstrate and verify ConditionalEstimator correctness."""

import numpy as np
from conditional_estimator import ConditionalEstimator


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ------------------------------------------------------------------ #
#  1. Perfect data — exact recovery                                   #
# ------------------------------------------------------------------ #
section("1. EXACT RECOVERY — noiseless linear data")

est = ConditionalEstimator()
# True function: market_share = 100 - 20 * price
prices = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
shares = [100, 80, 60, 40, 20, 0]
est.update_batch(prices, shares)

print("  price  | true share | estimated mean | error")
print("  -------+------------+----------------+------")
for p, s in zip(prices, shares):
    m = est.mean(p)
    print(f"  {p:5.1f}  |   {s:6.1f}   |    {m:8.4f}      | {abs(m - s):.1e}")

# interpolation at midpoints
print("\n  Interpolation at midpoints:")
for p in [0.5, 1.5, 2.5, 3.5, 4.5]:
    true = 100 - 20 * p
    m = est.mean(p)
    print(f"  price={p:.1f}  true={true:.1f}  est={m:.4f}  err={abs(m-true):.1e}")

root1 = est.find_root(50.0)
print(f"\n  Root search: share=50% → price={root1:.8f}  (expected 2.5)")

var_mid = est.variance(2.5)
print(f"  Variance at price=2.5: {var_mid:.2e}  (expected ~0, no noise)")


# ------------------------------------------------------------------ #
#  2. Monotonicity enforcement — PAVA in action                       #
# ------------------------------------------------------------------ #
section("2. PAVA MONOTONICITY — non-monotone raw data made monotone")

est2 = ConditionalEstimator()
raw_prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
raw_shares = [90,  40,  70,  55,  30,  45,  10]  # clearly not monotone
est2.update_batch(raw_prices, raw_shares)

print("  Raw observations:")
for p, s in zip(raw_prices, raw_shares):
    print(f"    price={p:.0f}  share={s}")

print("\n  Isotonic fit (must be non-increasing):")
est2._ensure_fit()
for x, y in zip(est2._iso_xs, est2._iso_ys):
    print(f"    price={x:.0f}  fitted_share={y:.2f}")

diffs = np.diff(est2._iso_ys)
print(f"\n  Consecutive differences: {np.round(diffs, 4)}")
print(f"  All non-increasing? {np.all(diffs <= 1e-12)}")

# dense grid check
grid = np.linspace(1, 7, 500)
means = [est2.mean(x) for x in grid]
grid_diffs = np.diff(means)
print(f"  Dense grid (500 pts) all non-increasing? {np.all(np.array(grid_diffs) <= 1e-12)}")


# ------------------------------------------------------------------ #
#  3. Noisy data — statistical properties                             #
# ------------------------------------------------------------------ #
section("3. NOISY DATA — true function 100-10x, noise σ=5")

rng = np.random.default_rng(42)
est3 = ConditionalEstimator(bandwidth=1.5)
n = 200
xs = rng.uniform(0, 10, n)
ys = 100 - 10 * xs + rng.normal(0, 5, n)
est3.update_batch(xs, ys)

print("  price | true E[Y] | estimated mean | abs error | est. std")
print("  ------+-----------+----------------+-----------+---------")
for p in [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    true_mean = 100 - 10 * p
    m = est3.mean(p)
    s = est3.std(p)
    print(f"  {p:5.1f} |  {true_mean:7.1f}   |    {m:8.2f}      |   {abs(m-true_mean):5.2f}    |  {s:.2f}")

root3 = est3.find_root(50.0)
print(f"\n  Root for share=50%: price={root3:.4f}  (true=5.0, error={abs(root3-5):.4f})")

# monotonicity on dense grid
grid = np.linspace(0, 10, 1000)
means = [est3.mean(x) for x in grid]
print(f"  Monotonicity on 1000-point grid: {np.all(np.diff(means) <= 1e-12)}")


# ------------------------------------------------------------------ #
#  4. Online updates — estimates evolve                               #
# ------------------------------------------------------------------ #
section("4. ONLINE UPDATES — estimates evolve as data arrives")

est4 = ConditionalEstimator()
print("  After each batch of observations, estimate mean at price=3.0:\n")

batches = [
    ([1.0, 5.0], [90, 10]),
    ([2.0, 4.0], [72, 28]),
    ([2.5, 3.5], [63, 38]),
    ([3.0, 3.0, 3.0], [52, 48, 50]),
]

for i, (bx, by) in enumerate(batches):
    est4.update_batch(bx, by)
    m, v = est4.mean_and_variance(3.0)
    print(f"  Batch {i+1}: added {list(zip(bx,by))}")
    print(f"           n_obs={est4.n_obs}  mean(3)={m:.2f}  std(3)={v**0.5:.2f}")

true_at_3 = 50.0  # by design the last batch centers around 50 at x=3
print(f"\n  Final mean(3.0)={est4.mean(3.0):.2f}  (data was designed to cluster around 50)")


# ------------------------------------------------------------------ #
#  5. Root finding accuracy across targets                            #
# ------------------------------------------------------------------ #
section("5. ROOT FINDING — sweep of targets on clean data")

est5 = ConditionalEstimator()
# dense noiseless data: share = 100 - 10*price
for p in np.linspace(0, 10, 101):
    est5.update(p, 100 - 10 * p)

print("  target share | found price | true price | error")
print("  -------------+-------------+------------+------")
for target in [90, 75, 50, 25, 10]:
    root = est5.find_root(target)
    true_p = (100 - target) / 10
    print(f"      {target:5.0f}     |   {root:8.5f}   |   {true_p:6.2f}    | {abs(root-true_p):.2e}")


# ------------------------------------------------------------------ #
#  6. Variance reflects local noise level (heteroscedastic)           #
# ------------------------------------------------------------------ #
section("6. HETEROSCEDASTIC NOISE — variance tracks local noise level")

rng = np.random.default_rng(7)
est6 = ConditionalEstimator(bandwidth=1.0)
# noise increases with x: σ(x) = 1 + 2*x
for _ in range(500):
    x = rng.uniform(0, 10)
    sigma = 1 + 2 * x
    y = 100 - 10 * x + rng.normal(0, sigma)
    est6.update(x, y)

print("  price | true σ | estimated std | ratio est/true")
print("  ------+--------+---------------+---------------")
for p in [1.0, 3.0, 5.0, 7.0, 9.0]:
    true_sigma = 1 + 2 * p
    est_std = est6.std(p)
    print(f"  {p:5.1f} | {true_sigma:5.1f}  |     {est_std:6.2f}    |    {est_std/true_sigma:.2f}")

print("\n  (Ratios should trend upward with x, showing variance tracks noise)")


# ------------------------------------------------------------------ #
#  7. Extrapolation is flat (constant beyond data)                    #
# ------------------------------------------------------------------ #
section("7. EXTRAPOLATION — flat beyond observed range")

est7 = ConditionalEstimator()
est7.update_batch([2, 4, 6, 8], [80, 60, 40, 20])

left_vals = [est7.mean(x) for x in [-5, -1, 0, 1, 2]]
right_vals = [est7.mean(x) for x in [8, 9, 10, 15, 100]]

print(f"  Leftmost observed: price=2, share=80")
print(f"  Left extrapolation:  mean at x=-5,-1,0,1,2 = {[round(v,1) for v in left_vals]}")
print(f"  All equal to 80? {all(abs(v - 80) < 1e-10 for v in left_vals[:-1])}")

print(f"\n  Rightmost observed: price=8, share=20")
print(f"  Right extrapolation: mean at x=8,9,10,15,100 = {[round(v,1) for v in right_vals]}")
print(f"  All equal to 20? {all(abs(v - 20) < 1e-10 for v in right_vals[1:])}")


# ------------------------------------------------------------------ #
#  Summary                                                            #
# ------------------------------------------------------------------ #
section("SUMMARY")
checks = [
    ("Exact recovery on noiseless data", abs(est.mean(2.5) - 50) < 1e-10),
    ("Root finding on noiseless data", abs(root1 - 2.5) < 1e-4),
    ("PAVA output is non-increasing", np.all(np.diff(est2._iso_ys) <= 1e-12)),
    ("Dense-grid monotonicity (noisy)", np.all(np.diff(means) <= 1e-12)),
    ("Noisy root within 0.5 of truth", abs(root3 - 5.0) < 0.5),
    ("Zero variance on noiseless data", est.variance(2.5) < 1e-10),
    ("Flat left extrapolation", abs(est7.mean(-10) - 80) < 1e-10),
    ("Flat right extrapolation", abs(est7.mean(100) - 20) < 1e-10),
    ("Online n_obs tracking", est4.n_obs == 9),
]

all_pass = True
for label, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] {label}")

print(f"\n  {'All checks passed!' if all_pass else 'SOME CHECKS FAILED'}")
