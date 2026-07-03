# Observations - run 027 `quiet-firebrand-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `bbrrydax`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=200; grad_skipped max=0; grad_has_nan max=0; grad_norm last=1.1283 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4799; dead_dim=0; cross_video_cosine=0.7359 |
| Q3 | Rich latent? | FAIL | c_effective_rank=8.8218; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=62.9978; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=3.3072; copy_loss=0.0525; ratio=62.9978; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=12.9475; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=1.0447; cplus=1.0451; chat=1.0439; chat-cplus=-0.0011; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0903 @ 0 | 2.6635 @ 200 | -0.4268 | 2.6635 / 3.0464 / 3.5373 | n/a |
| `L_flow` | 2.8744 @ 0 | 2.6065 @ 200 | -0.2679 | 2.6065 / 2.9448 / 3.3857 | n/a |
| `L_var` | 0.4318 @ 0 | 0.0788 @ 200 | -0.353 | 0.0433 / 0.1847 / 0.4318 | n/a |
| `L_recon` | 1.0438 @ 0 | 0.8786 @ 200 | -0.1652 | 0.8786 / 0.9649 / 1.0438 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 0.1 @ 200 | 0.1 | 0 / 0.05 / 0.1 | n/a |
| `c_std_mean` | 0.4799 @ 0 | 0.4799 @ 0 | 0 | 0.4799 / 0.4799 / 0.4799 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7359 @ 0 | 0.7359 @ 0 | 0 | 0.7359 / 0.7359 / 0.7359 | n/a |
| `c_effective_rank` | 8.8218 @ 0 | 8.8218 @ 0 | 0 | 8.8218 / 8.8218 / 8.8218 | n/a |
| `c_slot_diversity_rank` | 24.144 @ 0 | 24.144 @ 0 | 0 | 24.144 / 24.144 / 24.144 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 0 | 0 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9998 @ 0 | 0 | 0.9998 / 0.9998 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0525 @ 0 | 0.0525 @ 0 | 0 | 0.0525 / 0.0525 / 0.0525 | n/a |
| `coarse_model_loss` | 3.3072 @ 0 | 3.3072 @ 0 | 0 | 3.3072 / 3.3072 / 3.3072 | n/a |
| `coarse_batch_mean_loss` | 0.2554 @ 0 | 0.2554 @ 0 | 0 | 0.2554 / 0.2554 / 0.2554 | n/a |
| `coarse_vs_copy_ratio` | 62.9978 @ 0 | 62.9978 @ 0 | 0 | 62.9978 / 62.9978 / 62.9978 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.9475 @ 0 | 12.9475 @ 0 | 0 | 12.9475 / 12.9475 / 12.9475 | n/a |
| `L_recon_present` | 1.0447 @ 0 | 1.0447 @ 0 | 0 | 1.0447 / 1.0447 / 1.0447 | n/a |
| `L_recon_cplus` | 1.0451 @ 0 | 1.0451 @ 0 | 0 | 1.0451 / 1.0451 / 1.0451 | n/a |
| `L_recon_chat` | 1.0439 @ 0 | 1.0439 @ 0 | 0 | 1.0439 / 1.0439 / 1.0439 | n/a |
| `grad_norm` | 1.6122 @ 0 | 1.1283 @ 200 | -0.4839 | 0.507 / 1.0768 / 1.6122 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.134 @ 200 | 0.1333 | 6.667e-04 / 0.0673 / 0.134 | n/a |

## Last Key Metrics

`c_effective_rank`=8.8218 @ 0; `c_cross_video_cosine`=0.7359 @ 0; `c_std_mean`=0.4799 @ 0; `coarse_vs_copy_ratio`=62.9978 @ 0; `coarse_vs_batch_mean_ratio`=12.9475 @ 0; `L_recon_present`=1.0447 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 62.9978 and the latest batch-mean ratio is 12.9475; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 8.8218, cross-video cosine is 0.7359, and copy loss is 0.0525. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — quiet-firebrand-25 (combined "all bigger")

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **347 s**, last heartbeat **03:47:43Z** — same instant as all siblings
(synchronized whole-pod death; forensics in [Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0
(initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.045 | untrained (highest init of the wave — init variance, not signal) |
| `coarse_vs_copy_ratio` | 63.0 | random init |
| `c_effective_rank` | 8.82 | pre-training |
| `c_cross_video_cosine` | 0.736 | collapsed random init |

Nothing here is a result.

## What we still don't know

Whether the three levers compound (predicted: no — floor ~0.575–0.585, no break). Because this run is
confounded by design and its outcome is the most predictable in the wave, it is the **lowest re-run
priority** — see [NEXT_STEPS.md](NEXT_STEPS.md). The clean single-axis n_c runs
([pious-mountain-28](../run_026_pious-mountain-28/), [earnest-dragon-25](../run_025_earnest-dragon-25/)) carry the
decision; this run would only ever be a corroborating "best shot."
