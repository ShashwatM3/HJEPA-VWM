# Observations - run 016 `drawn-elevator-16`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `0n5mx3qf`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=finished; step=14950; grad_skipped max=1; grad_has_nan max=0; grad_norm last=191.3325 |
| Q2 | c_t alive / video-specific? | PASS | std=1.099; dead_dim=0; cross_video_cosine=0.167 |
| Q3 | Rich latent? | FAIL | c_effective_rank=13.621; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.2222 (0.2638 @ 7500 -> 0.486 @ 14500); ratio=3.3319; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.6192; copy_loss=0.486; ratio=3.3319; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.3557; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2746 @ 7500 | 1.5687 @ 14950 | 1.2942 | 0.2451 / 1.4604 / 1.9097 | 1.3226 / 1.6039 / 1.8796 (n=100) |
| `L_flow` | 0.2719 @ 7500 | 1.5681 @ 14950 | 1.2962 | 0.2402 / 1.4589 / 1.9088 | 1.3217 / 1.6027 / 1.8784 (n=100) |
| `L_var` | 0.0053 @ 7500 | 0.0013 @ 14950 | -0.004 | 9.693e-04 / 0.0031 / 0.0099 | 9.693e-04 / 0.0025 / 0.0048 (n=100) |
| `c_std_mean` | 1.0427 @ 7500 | 1.099 @ 14500 | 0.0564 | 1.0292 / 1.0878 / 1.1014 | 1.0979 / 1.0985 / 1.099 (n=10) |
| `c_dead_dim_frac` | 0 @ 7500 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.2423 @ 7500 | 0.167 @ 14500 | -0.0753 | 0.1662 / 0.1848 / 0.2799 | 0.167 / 0.1681 / 0.1693 (n=10) |
| `c_effective_rank` | 13.5749 @ 7500 | 13.621 @ 14500 | 0.0461 | 13.5749 / 13.6191 / 13.7355 | 13.5944 / 13.6121 / 13.621 (n=10) |
| `c_slot_diversity_rank` | 19.4345 @ 7500 | 18.5864 @ 14500 | -0.8481 | 18.5804 / 18.7677 / 19.4345 | 18.5804 / 18.6141 / 18.6957 (n=10) |
| `c_attn_entropy` | 0.9252 @ 7500 | 0.9171 @ 14500 | -0.0082 | 0.9171 / 0.9181 / 0.9252 | 0.9171 / 0.9173 / 0.9176 (n=10) |
| `c_attn_entropy_min` | 0.8025 @ 7500 | 0.7896 @ 14500 | -0.0129 | 0.7862 / 0.7904 / 0.8025 | 0.7895 / 0.7896 / 0.7898 (n=10) |
| `coarse_copy_loss` | 0.2638 @ 7500 | 0.486 @ 14500 | 0.2222 | 0.2638 / 0.4514 / 0.5078 | 0.4832 / 0.4973 / 0.5078 (n=10) |
| `coarse_model_loss` | 0.3519 @ 7500 | 1.6192 @ 14500 | 1.2674 | 0.2789 / 1.3386 / 1.7867 | 1.2778 / 1.4431 / 1.6292 (n=10) |
| `coarse_batch_mean_loss` | 1.1417 @ 7500 | 1.1944 @ 14500 | 0.0527 | 1.1343 / 1.1728 / 1.1944 | 1.1725 / 1.1818 / 1.1944 (n=10) |
| `coarse_vs_copy_ratio` | 1.3339 @ 7500 | 3.3319 @ 14500 | 1.9981 | 1.0344 / 2.8955 / 4.3489 | 2.5709 / 2.9041 / 3.3319 (n=10) |
| `coarse_vs_batch_mean_ratio` | 0.3082 @ 7500 | 1.3557 @ 14500 | 1.0475 | 0.2459 / 1.1379 / 1.5274 | 1.0737 / 1.2214 / 1.3863 (n=10) |
| `grad_norm` | 2.5287 @ 7500 | 191.3325 @ 14950 | 188.8038 | 2.2622 / 107.2735 / 366.5313 | 44.9707 / 130.3489 / 366.5313 (n=100) |
| `grad_skipped` | 0 @ 7500 | 1 @ 14950 | 1 | 0 / 0.8467 / 1 | 0 / 0.99 / 1 (n=100) |
| `grad_has_nan` | 0 @ 7500 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 0.5868 @ 7500 | 3.385e-05 @ 14950 | -0.5868 | 3.385e-05 / 0.2198 / 0.5868 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=13.621 @ 14500; `c_cross_video_cosine`=0.167 @ 14500; `c_std_mean`=1.099 @ 14500; `coarse_vs_copy_ratio`=3.3319 @ 14500; `coarse_vs_batch_mean_ratio`=1.3557 @ 14500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 3.3319 and the latest batch-mean ratio is 1.3557; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 13.621, cross-video cosine is 0.167, and copy loss is 0.486. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next pivot added reconstruction anchors to test whether c_t lacked usable information rather than just variance.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — drawn-elevator-16

**W&B:** [`0n5mx3qf`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0n5mx3qf) · state **finished** · runtime ~3h 29m · steps **7500–14950**

## Outcome

**Failed acceptance.** Reached 15k step count but **84.7% of logged steps skipped** (127/150).
Only ~23 optimizer updates post-resume. Grad-skip death spiral recurred from step **8550**.

## Key numbers

| When | Step | Highlights |
|---|---|---|
| Resume | 7500 | rank 13.57, copy 1.33, `grad_skipped=0` |
| Spike zone | 8500 | `grad_norm` 42.8, copy 4.35, still no skip |
| First skip | **8550** | `grad_norm` 62.5, `L_flow` 1.71 |
| Sustained skip | 8550–14950 | **127/150 rows skipped** |
| Final diag | 14500 | copy 3.33, rank **13.62**, std 1.10, cosine 0.17 |

## Interpretation

Halved coarse-flow LR (`1e-4`) **delayed** first skip by ~100 steps vs elated (8450) but did
**not** prevent the death spiral. Forward latents stayed deceptively healthy (rank ~13.6) —
same misleading pattern as elated post-break — because weights were frozen most of the time.

## vs royal-cherry-17

| | drawn | royal |
|---|---|---|
| AGC | off | on |
| `lr_coarse_flow` | 1e-4 | 2e-4 |
| Skip rate | 85% | **0%** |
| Final rank | 13.62 | **5.84** (collapsed) |
| Final copy ratio | 3.33 | **12.73** |

Drawn froze in a mediocre basin; royal kept learning into a worse one. Neither passes gates.
