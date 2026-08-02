# Observations - run 006 `exalted-lion-6`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `wv69n7n5`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=finished; step=450; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.666 |
| Q2 | c_t alive / video-specific? | PASS | std=0.8307; dead_dim=0; cross_video_cosine=0.2461 |
| Q3 | Rich latent? | FAIL | c_effective_rank=10.3413; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.4327 (0.0193 @ 0 -> 0.4519 @ 400); ratio=2.9663; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.3405; copy_loss=0.4519; ratio=2.9663; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.9614; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9153 @ 0 | 1.3908 @ 450 | -1.5245 | 1.3908 / 2.0852 / 3.4042 | n/a |
| `L_flow` | 2.8726 @ 0 | 1.3785 @ 450 | -1.4942 | 1.3785 / 2.0677 / 3.372 | n/a |
| `L_var` | 0.4267 @ 0 | 0.1234 @ 450 | -0.3033 | 0.0471 / 0.1749 / 0.4267 | n/a |
| `c_std_mean` | 0.4985 @ 0 | 0.8307 @ 400 | 0.3323 | 0.4985 / 0.7358 / 0.8307 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 400 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7174 @ 0 | 0.2461 @ 400 | -0.4713 | 0.2461 / 0.3806 / 0.7174 | n/a |
| `c_effective_rank` | 8.9986 @ 0 | 10.3413 @ 400 | 1.3427 | 5.991 / 8.3018 / 10.3413 | n/a |
| `c_slot_diversity_rank` | 16.1949 @ 0 | 16.6398 @ 400 | 0.4449 | 15.964 / 16.2544 / 16.6398 | n/a |
| `c_attn_entropy` | 1 @ 0 | 1 @ 400 | 0 | 1 / 1 / 1 | n/a |
| `coarse_copy_loss` | 0.0193 @ 0 | 0.4519 @ 400 | 0.4327 | 0.0193 / 0.3748 / 0.5955 | n/a |
| `coarse_model_loss` | 3.3071 @ 0 | 1.3405 @ 400 | -1.9666 | 1.3405 / 2.0869 / 3.3071 | n/a |
| `coarse_batch_mean_loss` | 0.27 @ 0 | 0.6835 @ 400 | 0.4135 | 0.27 / 0.4452 / 0.6835 | n/a |
| `coarse_vs_copy_ratio` | 171.5862 @ 0 | 2.9663 @ 400 | -168.62 | 2.4936 / 37.7955 / 171.5862 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.2494 @ 0 | 1.9614 @ 400 | -10.288 | 1.9614 / 5.8512 / 12.2494 | n/a |
| `grad_norm` | 0.3553 @ 0 | 2.666 @ 450 | 2.3107 | 0.1942 / 0.9116 / 2.666 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 450 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 400 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.3007 @ 450 | 0.3 | 6.667e-04 / 0.1507 / 0.3007 | n/a |

## Last Key Metrics

`c_effective_rank`=10.3413 @ 400; `c_cross_video_cosine`=0.2461 @ 400; `c_std_mean`=0.8307 @ 400; `coarse_vs_copy_ratio`=2.9663 @ 400; `coarse_vs_batch_mean_ratio`=1.9614 @ 400

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.9663 and the latest batch-mean ratio is 1.9614; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 10.3413, cross-video cosine is 0.2461, and copy loss is 0.4519. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

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
[`sleek-leaf-7`](../run_007_sleek-leaf-7/) on full SSv2 with the *fixed per-head* metric,
where `c_slot_diversity_rank` read **1.62/32**. So this run did not itself motivate the
slot loss — it motivated VICReg-C and proved init is a non-lever.

## Interpretation

- Collapse is architectural/objective, not data-scale or init.
- This run becomes the clean instrumented baseline; the next move is full SSv2 with the
  per-head entropy fix and `lambda_cov` logged at 0 for calibration
  ([`sleek-leaf-7`](../run_007_sleek-leaf-7/)).
