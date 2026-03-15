"""Visual proof that ConditionalEstimator works."""

import numpy as np
import matplotlib.pyplot as plt
from conditional_estimator import ConditionalEstimator


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ConditionalEstimator — Visual Proof", fontsize=15, fontweight="bold")


# ------------------------------------------------------------------ #
#  Panel 1: Noiseless — exact recovery + root finding                 #
# ------------------------------------------------------------------ #
ax = axes[0, 0]
ax.set_title("1. Exact recovery + root finding (noiseless)")

est = ConditionalEstimator()
prices = [0, 1, 2, 3, 4, 5]
shares = [100, 80, 60, 40, 20, 0]
est.update_batch(prices, shares)

grid = np.linspace(-0.5, 5.5, 300)
means = [est.mean(x) for x in grid]

ax.scatter(prices, shares, c="black", zorder=5, label="observations", s=60)
ax.plot(grid, means, "b-", lw=2, label="estimated mean")
ax.plot(grid, [100 - 20 * x for x in grid], "r--", lw=1, alpha=0.6, label="true function")

root = est.find_root(50.0)
ax.axhline(50, color="gray", ls=":", lw=1)
ax.axvline(root, color="green", ls="--", lw=1.5)
ax.plot(root, 50, "g*", ms=15, zorder=6, label=f"root at price={root:.2f}")

ax.set_xlabel("Price")
ax.set_ylabel("Market Share (%)")
ax.legend(fontsize=8)
ax.set_ylim(-10, 115)


# ------------------------------------------------------------------ #
#  Panel 2: PAVA monotonicity on messy data                           #
# ------------------------------------------------------------------ #
ax = axes[0, 1]
ax.set_title("2. PAVA enforces monotonicity (non-monotone input)")

est2 = ConditionalEstimator()
raw_p = [1, 2, 3, 4, 5, 6, 7]
raw_s = [90, 40, 70, 55, 30, 45, 10]
est2.update_batch(raw_p, raw_s)

grid2 = np.linspace(0.5, 7.5, 300)
means2 = [est2.mean(x) for x in grid2]

ax.scatter(raw_p, raw_s, c="red", marker="x", s=80, zorder=5, label="raw (non-monotone)")
ax.plot(grid2, means2, "b-", lw=2, label="isotonic fit")

est2._ensure_fit()
ax.step(est2._iso_xs, est2._iso_ys, where="mid", color="blue", alpha=0.3, lw=6, label="PAVA steps")

ax.set_xlabel("Price")
ax.set_ylabel("Market Share (%)")
ax.legend(fontsize=8)


# ------------------------------------------------------------------ #
#  Panel 3: Noisy data — mean + confidence band + root                #
# ------------------------------------------------------------------ #
ax = axes[1, 0]
ax.set_title("3. Noisy data: mean ± 2σ band + root finding")

rng = np.random.default_rng(42)
est3 = ConditionalEstimator(bandwidth=1.5)
n = 200
xs = rng.uniform(0, 10, n)
ys = 100 - 10 * xs + rng.normal(0, 5, n)
est3.update_batch(xs, ys)

grid3 = np.linspace(0, 10, 400)
means3 = np.array([est3.mean(x) for x in grid3])
stds3 = np.array([est3.std(x) for x in grid3])

ax.scatter(xs, ys, c="gray", alpha=0.3, s=10, label="observations")
ax.plot(grid3, means3, "b-", lw=2, label="estimated mean")
ax.fill_between(grid3, means3 - 2 * stds3, means3 + 2 * stds3,
                color="blue", alpha=0.15, label="±2σ band")
ax.plot(grid3, 100 - 10 * grid3, "r--", lw=1, alpha=0.6, label="true E[Y|X]")

root3 = est3.find_root(50.0)
ax.axhline(50, color="gray", ls=":", lw=1)
ax.axvline(root3, color="green", ls="--", lw=1.5)
ax.plot(root3, 50, "g*", ms=15, zorder=6, label=f"root={root3:.2f} (true=5.0)")

ax.set_xlabel("Price")
ax.set_ylabel("Market Share (%)")
ax.legend(fontsize=8, loc="upper right")


# ------------------------------------------------------------------ #
#  Panel 4: Online updates — estimate evolution                       #
# ------------------------------------------------------------------ #
ax = axes[1, 1]
ax.set_title("4. Online updates: estimate evolves with data")

colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
batches = [
    ([1.0, 9.0], [92, 8]),
    ([3.0, 7.0], [72, 28]),
    ([4.0, 6.0], [58, 42]),
    ([2.0, 5.0, 8.0], [82, 50, 18]),
]

est4 = ConditionalEstimator()
grid4 = np.linspace(0, 10, 300)

for i, (bx, by) in enumerate(batches):
    est4.update_batch(bx, by)
    m4 = [est4.mean(x) for x in grid4]
    ax.plot(grid4, m4, color=colors[i], lw=1.5, alpha=0.7,
            label=f"after batch {i+1} (n={est4.n_obs})")
    ax.scatter(bx, by, color=colors[i], s=40, zorder=5, edgecolors="black", linewidths=0.5)

# final root
root4 = est4.find_root(50.0)
ax.axhline(50, color="gray", ls=":", lw=1)
ax.axvline(root4, color="green", ls="--", lw=1.5)
ax.plot(root4, 50, "g*", ms=15, zorder=6, label=f"final root={root4:.2f}")

ax.set_xlabel("Price")
ax.set_ylabel("Market Share (%)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to proof.png")
