"""Plot ConditionalEstimator on user-provided observations.

Shows before/after comparison of the three improvements:
  1. Weighted PAVA (cluster resists outlier)
  2. Uncertainty-aware variance (honest at sparse regions)
  3. Bootstrap CI on root
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import Counter
from conditional_estimator import ConditionalEstimator

# User observations: (parameter, target)
observations = [
    (0, 100),
    (-10, 50),
    (-10, 60),
    (-9, 55),
    (-9, 53),
    (-9, 48),
    (20, 30),
    (10, 40),
    (-100, 100),
]

xs = [o[0] for o in observations]
ys = [o[1] for o in observations]

est = ConditionalEstimator(bandwidth=5.0)
est.update_batch(xs, ys)
est._ensure_fit()

x_min, x_max = min(xs) - 10, max(xs) + 10
grid = np.linspace(x_min, x_max, 500)
means = np.array([est.mean(x) for x in grid])
stds = np.array([est.std(x) for x in grid])

# ---- Build old (unweighted) PAVA for comparison ----
xs_arr, ys_arr = np.array(xs), np.array(ys)
order = np.argsort(xs_arr)
ux, inv = np.unique(xs_arr[order], return_inverse=True)
uy = np.zeros_like(ux, dtype=float)
counts = np.zeros_like(ux, dtype=float)
np.add.at(uy, inv, ys_arr[order])
np.add.at(counts, inv, 1.0)
uy /= counts
n = len(uy)
bv, bw, bi = list(uy), [1.0]*n, list(range(n))
i = 0
while i < len(bv) - 1:
    if bv[i] < bv[i+1]:
        w = bw[i]+bw[i+1]
        bv[i] = (bv[i]*bw[i]+bv[i+1]*bw[i+1])/w
        bw[i] = w
        del bv[i+1], bw[i+1], bi[i+1]
        if i > 0: i -= 1
    else:
        i += 1
old_iy = np.empty(n)
for k in range(len(bi)):
    s = bi[k]
    e = bi[k+1] if k+1 < len(bi) else n
    old_iy[s:e] = bv[k]
old_f = interp1d(ux, old_iy, kind="linear",
                 fill_value=(old_iy[0], old_iy[-1]), bounds_error=False)
old_means = old_f(grid)

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ConditionalEstimator — User Data (improved)", fontsize=15, fontweight="bold")

# Panel 1: Weighted PAVA before vs after
ax = axes[0, 0]
ax.set_title("1. Weighted PAVA: plateau 69 → 61")
ax.scatter(xs, ys, c="black", s=60, zorder=5, label="observations")
ax.plot(grid, means, "b-", lw=2, label="new (weighted)")
ax.plot(grid, old_means, "r--", lw=1.5, alpha=0.6, label="old (unweighted)")
ax.annotate("old: 69", (-5, 69), fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"), xytext=(-30, 78))
ax.annotate("new: 61", (-5, 61), fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"), xytext=(-30, 50))
ax.set_xlabel("Parameter")
ax.set_ylabel("Target")
ax.legend(fontsize=8)

# Panel 2: Variance with σ curve
ax = axes[0, 1]
ax.set_title("2. Variance: honest at sparse regions")
ax.scatter(xs, ys, c="gray", alpha=0.4, s=30)
ax.plot(grid, means, "b-", lw=2)
ax.fill_between(grid, means - 2*stds, means + 2*stds,
                color="blue", alpha=0.15, label="±2σ (uncertainty-aware)")
ax2 = ax.twinx()
ax2.plot(grid, stds, "g-", lw=1.5, alpha=0.8, label="σ(x)")
ax2.set_ylabel("σ(x)", color="green")
ax2.tick_params(axis="y", labelcolor="green")
ax.set_xlabel("Parameter")
ax.set_ylabel("Target")
ax.legend(fontsize=8, loc="upper right")
ax2.legend(fontsize=8, loc="center right")

# Panel 3: Root finding with bootstrap CI
ax = axes[1, 0]
ax.set_title("3. Root finding with 95% bootstrap CI")
ax.scatter(xs, ys, c="gray", alpha=0.5, s=40, zorder=3)
ax.plot(grid, means, "b-", lw=2, label="estimated mean")
ax.fill_between(grid, means - 2*stds, means + 2*stds, color="blue", alpha=0.1)
targets = [50, 60, 70, 80]
colors_t = plt.cm.viridis(np.linspace(0.2, 0.9, len(targets)))
for target, c in zip(targets, colors_t):
    try:
        root, ci_lo, ci_hi = est.find_root_ci(target, rng=42)
        ax.axhline(target, color=c, ls=":", lw=0.8, alpha=0.6)
        ax.plot(root, target, "*", color=c, ms=14, zorder=6)
        ax.plot([ci_lo, ci_hi], [target, target], "-", color=c, lw=3, alpha=0.4)
        ax.plot([ci_lo, ci_hi], [target, target], "|", color=c, ms=10)
        ax.annotate(f"{target}%: {root:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]",
                    (root, target), textcoords="offset points",
                    xytext=(10, 8), fontsize=7, color=c)
    except ValueError as e:
        print(f"  target={target}% — {e}")
ax.set_xlabel("Parameter")
ax.set_ylabel("Target")

# Panel 4: Isotonic fit detail with obs counts
ax = axes[1, 1]
ax.set_title("4. Isotonic fit detail (obs counts shown)")
ax.scatter(xs, ys, c="red", marker="x", s=80, zorder=5, label="raw observations")
ax.step(est._iso_xs, est._iso_ys, where="mid", color="blue", lw=4, alpha=0.3, label="PAVA steps")
ax.plot(grid, means, "b-", lw=2, label="interpolated fit")
for x_iso, y_iso in zip(est._iso_xs, est._iso_ys):
    ax.annotate(f"{y_iso:.0f}", (x_iso, y_iso), textcoords="offset points",
                xytext=(5, 8), fontsize=9, color="blue", fontweight="bold")
x_counts = Counter(xs)
for x_val, cnt in x_counts.items():
    if cnt > 1:
        ax.annotate(f"n={cnt}", (x_val, min(y for xo, y in zip(xs, ys) if xo == x_val)),
                    textcoords="offset points", xytext=(5, -15), fontsize=8,
                    color="darkred", fontstyle="italic")
ax.set_xlabel("Parameter")
ax.set_ylabel("Target")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("user_data_proof.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Summary table ----
print("\n  Isotonic fit (weighted PAVA):")
print(f"  {'param':>8s}  {'fitted':>8s}")
for x_iso, y_iso in zip(est._iso_xs, est._iso_ys):
    print(f"  {x_iso:8.1f}  {y_iso:8.2f}")

print(f"\n  Root search with 95% bootstrap CI:")
for t in [40, 50, 60, 70, 80, 90]:
    try:
        r, lo, hi = est.find_root_ci(t, rng=42)
        print(f"    target={t}%  →  x={r:7.2f}  95% CI [{lo:7.2f}, {hi:7.2f}]")
    except ValueError as e:
        print(f"    target={t}%  →  {e}")

print("\nSaved to user_data_proof.png")
