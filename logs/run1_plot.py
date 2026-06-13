"""Plot Run 1 (peachy-terrain-5) trajectory from the data points we have.

Data sources:
- tmux pastes from the live training session (steps 0, 500, 1000, 8000, 8500, 10500)
- W&B Summary tab screenshot at ~step 9500 (lr_mult=0.965)
- The explosion sequence at steps 10550, 10600, 10650, 10700, 10750+

Coverage gap: we are missing the diagnostic batches between step 1500 and 7500
(the user never pasted them). The W&B run has them but the chart UI didn't render.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Diagnostic-batch snapshots (logged every 500 steps).
diag_step = [0, 500, 1000, 8000, 8500, 9500, 10500]

c_effective_rank = [9.00, 5.29, 10.48, 4.38, 4.97, 5.30, 4.90]
c_cross_video_cosine = [0.718, 0.371, 0.307, 0.297, 0.431, 0.416, 0.268]
c_std_mean = [0.498, 0.748, 0.799, 0.826, 0.747, 0.761, 0.861]
c_std_median = [0.485, 0.742, 0.791, 0.844, 0.734, 0.732, 0.835]
coarse_vs_copy_ratio = [171.6, 3.27, 12.93, 5.01, 2.29, 7.90, 7.94]
coarse_vs_batch_mean_ratio = [12.25, 2.60, 1.74, 1.33, 1.30, 1.44, 1.27]
L_flow_diag = [2.87, 1.88, 1.25, 1.07, 1.04, 1.085, 1.07]
L_var_diag = [0.43, 0.07, 0.12, 0.045, 0.138, 0.053, 0.042]

# Pre-clip grad norm in the explosion window (per-step, not per-diag).
explosion_step = [10500, 10550, 10600, 10650, 10700, 10750]
explosion_grad_norm = [3.72, 864.0, 406.0, 2.6e8, 3.5e16, float("nan")]

# Phase 1 acceptance targets (30k spec).
TARGETS_30K = {
    "coarse_vs_copy_ratio": 0.70,
    "coarse_vs_batch_mean_ratio": 0.50,
    "c_effective_rank": 60,
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "Run 1 (peachy-terrain-5) — diagnostic trajectory through the crash",
    fontsize=13,
    fontweight="bold",
)

# Panel 1: effective rank
ax = axes[0, 0]
ax.plot(diag_step, c_effective_rank, "o-", color="#cc4444", linewidth=2, markersize=7)
ax.axhline(TARGETS_30K["c_effective_rank"], color="grey", linestyle="--", alpha=0.5)
ax.text(
    250,
    TARGETS_30K["c_effective_rank"] + 2,
    "30k spec target: > 60",
    fontsize=8,
    color="grey",
)
ax.set_title("c_effective_rank  (concern: stuck at ~5)", fontsize=10)
ax.set_xlabel("step")
ax.set_ylabel("rank")
ax.set_ylim(0, 70)
ax.grid(alpha=0.3)

# Panel 2: cross-video cosine
ax = axes[0, 1]
ax.plot(
    diag_step, c_cross_video_cosine, "o-", color="#4477aa", linewidth=2, markersize=7
)
ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
ax.text(250, 0.52, "spec: well < 0.5", fontsize=8, color="grey")
ax.set_title("c_cross_video_cosine  (healthy)", fontsize=10)
ax.set_xlabel("step")
ax.set_ylabel("mean pairwise cosine")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# Panel 3: variance (std mean + median)
ax = axes[0, 2]
ax.plot(diag_step, c_std_mean, "o-", color="#44aa66", linewidth=2, label="mean")
ax.plot(
    diag_step,
    c_std_median,
    "s--",
    color="#88cc88",
    linewidth=1.5,
    label="median",
    markersize=5,
)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.text(250, 1.02, "var-floor target: std → 1.0", fontsize=8, color="grey")
ax.set_title("c_std_mean / c_std_median  (healthy)", fontsize=10)
ax.set_xlabel("step")
ax.set_ylabel("per-dim std")
ax.set_ylim(0, 1.2)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# Panel 4: vs copy ratio
ax = axes[1, 0]
ax.semilogy(
    diag_step, coarse_vs_copy_ratio, "o-", color="#cc4444", linewidth=2, markersize=7
)
ax.axhline(
    TARGETS_30K["coarse_vs_copy_ratio"], color="grey", linestyle="--", alpha=0.5
)
ax.text(250, 0.85, "30k spec target: ≤ 0.70", fontsize=8, color="grey")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.4)
ax.text(
    250, 1.05, "1.0 = parity with 'copy context'", fontsize=8, color="black", alpha=0.7
)
# Highlight the step-8500 best
ax.annotate(
    "best: 2.29 at step 8500",
    xy=(8500, 2.29),
    xytext=(5000, 8),
    fontsize=8,
    color="#882222",
    arrowprops=dict(arrowstyle="->", color="#882222", lw=1.0),
)
ax.set_title("coarse_vs_copy_ratio  (oscillating)", fontsize=10)
ax.set_xlabel("step")
ax.set_ylabel("ratio (log scale)")
ax.grid(alpha=0.3, which="both")

# Panel 5: vs batch-mean ratio
ax = axes[1, 1]
ax.plot(
    diag_step,
    coarse_vs_batch_mean_ratio,
    "o-",
    color="#4477aa",
    linewidth=2,
    markersize=7,
)
ax.axhline(
    TARGETS_30K["coarse_vs_batch_mean_ratio"],
    color="grey",
    linestyle="--",
    alpha=0.5,
)
ax.text(250, 0.55, "30k spec target: ≤ 0.50", fontsize=8, color="grey")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.4)
ax.text(
    250, 1.02, "1.0 = parity with 'predict the mean'", fontsize=8, color="black", alpha=0.7
)
ax.set_title("coarse_vs_batch_mean_ratio  (slowly improving)", fontsize=10)
ax.set_xlabel("step")
ax.set_ylabel("ratio")
ax.set_ylim(0, max(coarse_vs_batch_mean_ratio) * 1.2)
ax.grid(alpha=0.3)

# Panel 6: pre-clip grad norm explosion (log y)
ax = axes[1, 2]
# Plot the explosion window with markers + connecting line, skipping NaN
xs = [s for s, g in zip(explosion_step, explosion_grad_norm) if g == g]  # NaN-filter
ys = [g for g in explosion_grad_norm if g == g]
ax.semilogy(xs, ys, "o-", color="#cc4444", linewidth=2, markersize=8)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.text(10495, 1.5, "Run 1 clip threshold = 1.0", fontsize=8, color="grey")
ax.axhline(0.5, color="green", linestyle="--", alpha=0.5)
ax.text(10495, 0.55, "Run 2 clip threshold = 0.5", fontsize=8, color="green")
ax.axhline(50, color="orange", linestyle="--", alpha=0.5)
ax.text(10495, 60, "Run 2 skip threshold = 50", fontsize=8, color="orange")
# Annotate NaN
ax.annotate(
    "NaN at step 10750",
    xy=(10750, 1e18),
    xytext=(10580, 1e15),
    fontsize=8,
    color="#882222",
    arrowprops=dict(arrowstyle="->", color="#882222", lw=1.0),
)
ax.set_title("Pre-clip grad_norm  (the explosion)", fontsize=10)
ax.set_xlabel("step")
ax.set_ylabel("pre-clip norm (log scale)")
ax.set_ylim(0.1, 1e20)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("logs/run1_trajectory.png", dpi=140, bbox_inches="tight")
print("Saved logs/run1_trajectory.png")
