"""One-shot plotting helper for logs/1.json. Saves PNGs alongside the JSON."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).parent
with open(LOG_DIR / "1.json") as f:
    data = json.load(f)

steps = [row["step"] for row in data["steps"]]
L_flow = [row["L_flow"] for row in data["steps"]]
L_var = [row["L_var"] for row in data["steps"]]
loss = [row["loss"] for row in data["steps"]]
grad_norm = [row["grad_norm"] for row in data["steps"]]
lr_mult = [row["lr_mult"] for row in data["steps"]]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Phase 1 smoke run — 200 steps, ssv2_tiny", fontsize=13)

ax = axes[0, 0]
ax.plot(steps, L_flow, color="#1f77b4", linewidth=1)
ax.set_title("L_flow  (flow-matching MSE; train batch)")
ax.set_xlabel("step")
ax.set_ylabel("L_flow")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(steps, L_var, color="#d62728", linewidth=1.2)
ax.set_title("L_var  (variance floor on c_t; train batch)")
ax.set_xlabel("step")
ax.set_ylabel("L_var")
ax.grid(alpha=0.3)
ax.annotate(
    "c_t spreading\n(healthy)",
    xy=(175, L_var[175]),
    xytext=(80, 0.30),
    arrowprops=dict(arrowstyle="->", color="grey"),
    fontsize=9,
    color="grey",
)

ax = axes[1, 0]
ax.plot(steps, loss, color="#2ca02c", linewidth=1, label="loss = L_flow + 0.10·L_var")
ax.plot(steps, grad_norm, color="#ff7f0e", linewidth=1, label="grad_norm")
ax.set_title("Total loss + gradient norm")
ax.set_xlabel("step")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

ax = axes[1, 1]
ax.plot(steps, lr_mult, color="#9467bd", linewidth=1.5)
ax.set_title("LR schedule multiplier  (linear warmup, 10k steps)")
ax.set_xlabel("step")
ax.set_ylabel("lr_mult")
ax.grid(alpha=0.3)
ax.annotate(
    f"end at step 199:\nlr_mult={lr_mult[-1]:.3f}\n(2% of full LR)",
    xy=(199, lr_mult[-1]),
    xytext=(60, 0.012),
    arrowprops=dict(arrowstyle="->", color="grey"),
    fontsize=9,
    color="grey",
)

plt.tight_layout()
out = LOG_DIR / "1_metrics.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
