# Observations — exalted-lion-6

## Outcome

**Diagnostic success.** Instrumentation works on the real pipeline, training is
healthy, and **Run-1's rank collapse is reproduced and confirmed persistent** (not an
init/early transient). At the time, this run's verdict pointed the team toward
**VICReg-C / feature decorrelation** — *not* slot loss (see "What this did NOT yet
show").

## Key numbers (step 0 → 450, verified vs W&B `wv69n7n5`)

| Metric | step 0 | step 100 | step 300 | step 400 | Reading |
|---|---|---|---|---|---|
| `L_flow` | 2.87 | 2.82 | 1.75 | 1.40 | learning well |
| `c_effective_rank` | 9.0 | 8.7 | 6.0 | 10.3 | **bouncing ~6–10, not climbing** |
| `c_slot_diversity_rank` | 16.2 | 16.0 | 16.3 | 16.6 | flat ~16/32 (with the head-averaged-era metric) |
| `c_cross_video_cosine` | 0.72 | 0.33 | 0.31 | 0.25 | videos becoming distinguishable |
| `c_std_mean` | 0.50 | 0.77 | 0.79 | 0.83 | rising toward the 1.0 floor — alive |
| `c_attn_entropy` | 0.99999 | 0.99999 | 0.99999 | 0.99999 | pinned uniform (head-averaged — flawed) |
| `coarse_vs_copy_ratio` | 171 | 8.6 | 2.5 | 3.0 | 171 at init is an artifact (EMA target ≈ online `c_t`) |
| `grad_skipped` | 0 | 0 | 0 | 0 | clean |

## Key finding (as read at the time)

- **Collapse is a persistent objective-level failure.** Rank bounces ~8 and does not
  climb, even while everything else (variance, cross-video distinctness, loss) trains
  fine. Videos are distinguishable (cosine 0.25) yet live in ~8 shared directions →
  textbook dimensional collapse.
- **Init knobs are a non-lever.** Rank/attention did not depend on the
  `small_gaussian`/scale/`out_mlp` init settings. This is exactly the result that led
  to commit `62b94dd` baking a fixed default and **deleting** the init flags ("fixes
  are baked in; only empirical knobs stay as switches").
- **An instrumentation flaw was caught here:** `c_attn_entropy = 0.9999994` (uniform)
  at *every* tick contradicts `c_slot_diversity_rank ≈ 16/32`. If attention were truly
  uniform, all 32 slots would read the mean token and slot-rank would be ~1, not 16.
  Cause: the metric used PyTorch's **head-averaged** attention weights
  (`average_attn_weights=True`), so 8 sharp-but-different per-head distributions
  average to look flat. The fix (per-head `c_attn_entropy_min`) was bundled into
  `a96c0d6` for the next run.

## What this did NOT yet show (temporal correction)

Because the entropy metric was head-averaged and `slot_diversity_rank` here read ~16,
the exalted-era conclusion was **"attention saturation is probably NOT the bottleneck;
feature correlation is → VICReg-C (P2) is the right next move."** The **slot-collapse**
signal that triggered the slot-loss arc came **later**, from
[`sleek-leaf-7`](../sleek-leaf-7/) on full SSv2 with the *fixed per-head* metric,
where `c_slot_diversity_rank` read **1.62/32**. So this run did not itself motivate the
slot loss — it motivated VICReg-C and proved init is a non-lever.

## Interpretation

- Collapse is architectural/objective, not data-scale or init.
- This run becomes the clean instrumented baseline; the next move is full SSv2 with the
  per-head entropy fix and `lambda_cov` logged at 0 for calibration
  ([`sleek-leaf-7`](../sleek-leaf-7/)).
