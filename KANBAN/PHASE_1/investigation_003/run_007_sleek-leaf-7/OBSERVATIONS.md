# Observations - run 007 `sleek-leaf-7`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `rpxyg9qt`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=crashed; step=4450; grad_skipped max=1; grad_has_nan max=0; grad_norm last=14.0997 |
| Q2 | c_t alive / video-specific? | PARTIAL | std=0.6909; dead_dim=0; cross_video_cosine=0.4996 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.4703; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.444 (0.0183 @ 0 -> 0.4623 @ 4000); ratio=2.3479; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.0855; copy_loss=0.4623; ratio=2.3479; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.3945; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9109 @ 0 | 1.1385 @ 4450 | -1.7724 | 1.0959 / 1.2795 / 3.4033 | n/a |
| `L_flow` | 2.8667 @ 0 | 1.1311 @ 4450 | -1.7356 | 1.0918 / 1.2649 / 3.3732 | n/a |
| `L_var` | 0.4419 @ 0 | 0.0744 @ 4450 | -0.3675 | 0.0405 / 0.1455 / 0.4419 | n/a |
| `c_std_mean` | 0.4958 @ 0 | 0.6909 @ 4000 | 0.1951 | 0.4958 / 0.689 / 0.8449 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 4000 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7217 @ 0 | 0.4996 @ 4000 | -0.2222 | 0.2491 / 0.4768 / 0.7217 | n/a |
| `c_effective_rank` | 9.2469 @ 0 | 9.4703 @ 4000 | 0.2233 | 6.9598 / 8.8843 / 10.6107 | n/a |
| `c_slot_diversity_rank` | 16.614 @ 0 | 1.5971 @ 4000 | -15.0169 | 1.5971 / 6.9679 / 18.4146 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9945 @ 4000 | -0.0055 | 0.9944 / 0.9982 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9688 @ 4000 | -0.0311 | 0.9563 / 0.9822 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0183 @ 0 | 0.4623 @ 4000 | 0.444 | 0.0183 / 0.2549 / 0.4623 | n/a |
| `coarse_model_loss` | 3.3121 @ 0 | 1.0855 @ 4000 | -2.2266 | 1.0855 / 1.3822 / 3.3121 | n/a |
| `coarse_batch_mean_loss` | 0.259 @ 0 | 0.7784 @ 4000 | 0.5194 | 0.259 / 0.6052 / 0.7784 | n/a |
| `coarse_vs_copy_ratio` | 180.5897 @ 0 | 2.3479 @ 4000 | -178.2418 | 2.3479 / 24.0454 / 180.5897 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.7899 @ 0 | 1.3945 @ 4000 | -11.3954 | 1.3945 / 3.0028 / 12.7899 | n/a |
| `grad_norm` | 0.3527 @ 0 | 14.0997 @ 4450 | 13.747 | 0.2019 / 24.6873 / 463.198 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 4450 | 0 | 0 / 0.1111 / 1 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 4000 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.8867 @ 4450 | 0.8861 | 6.667e-04 / 0.8022 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=9.4703 @ 4000; `c_cross_video_cosine`=0.4996 @ 4000; `c_std_mean`=0.6909 @ 4000; `coarse_vs_copy_ratio`=2.3479 @ 4000; `coarse_vs_batch_mean_ratio`=1.3945 @ 4000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.3479 and the latest batch-mean ratio is 1.3945; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.4703, cross-video cosine is 0.4996, and copy loss is 0.4623. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — sleek-leaf-7

## Outcome

**Negative for collapse fix.** Init + full data helped but did not solve dimensional
collapse. Run stopped early (~step 3500) once the data-confound and calibration
questions were answered (VICReg path **Run A**).

## Step-3500 snapshot (chat record)

Full SSv2, `lambda_cov=0` (logs `L_cov` only), per-head entropy fix active:

| Metric | Value | Healthy target |
|---|---|---|
| `c_effective_rank` | **8.7** / 256 | >60 |
| `c_slot_diversity_rank` | **1.62** / 32 | →32 |
| `c_attn_entropy_min` | **0.96** | low |
| `c_cross_video_cosine` | 0.25 | <0.5 |
| `c_std_mean` | 0.84 | ~1.0 |
| `coarse_vs_copy_ratio` | **7.42** | <1.0 |
| `L_flow` / `L_cov` | 1.10 / 20.18 | — (calibration) |

## Interpretation

- **Not data-limited:** 42× more data (tiny → full 169k) moved rank ~5 → ~8.7 only
- **Dominant failure:** slot redundancy + uniform attention (per pre-registered diagnostic rule)
- VICReg-C targets cross-video correlation (rank 8.7) but **not** within-video slot collapse
- `λ_cov` calibration ready: start **0.0027** (5% of `L_flow`)

Same physical run as BRIEF "Run 2" and VICReg "Run A". No W&B display name in chat.

Source: `COMPLETE_FULL_CHAT` lines ~6200–6241, ~9268–9271.

## Surprises

BRIEF summary cited `slot_diversity_rank ~16` — that reflected pre–per-head-fix
readings (`exalted-lion-6` era). With fixed metrics on full data, slot collapse
(1.62) was the louder signal.
