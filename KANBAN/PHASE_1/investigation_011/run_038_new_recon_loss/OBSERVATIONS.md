# Observations - run 038 `new_recon_loss`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `1u69hpfm`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Healthy rep, no predictor**  
**Verdict note:** Representation health is not the limiting issue; F_c still does not beat copy.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | PASS | state=finished; step=14950; grad_skipped max=0; grad_has_nan max=0; grad_norm last=1.8967 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0135; dead_dim=0; cross_video_cosine=0.1469 |
| Q3 | Rich latent? | PASS | c_effective_rank=60.3506; c_plus_effective_rank=60.6463 |
| Q4 | Temporal dynamics? | PARTIAL | copy_loss trend=up 1.5794 (0.0533 @ 0 -> 1.6327 @ 14500); ratio=1.0575; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.7266; copy_loss=1.6327; ratio=1.0575; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.1463; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.3459; cplus=0.3438; chat=0.3533; chat-cplus=0.0095; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Healthy rep, no predictor | Representation health is not the limiting issue; F_c still does not beat copy. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 1.3752 @ 0 | 1.7516 @ 14950 | 0.3764 | 0.4203 / 1.3755 / 1.9224 | 1.6392 / 1.7864 / 1.9224 (n=100) |
| `L_flow` | 1.1691 @ 0 | 1.694 @ 14950 | 0.5249 | 0.3418 / 1.3125 / 1.8633 | 1.5786 / 1.7285 / 1.8633 (n=100) |
| `L_var` | 0.4121 @ 0 | 0.0154 @ 14950 | -0.3967 | 0.014 / 0.0222 / 0.4121 | 0.014 / 0.0163 / 0.0184 (n=100) |
| `L_sigreg` | 0.0444 @ 0 | 0.0024 @ 14950 | -0.042 | 0.0017 / 0.0038 / 0.0444 | 0.0017 / 0.0024 / 0.003 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `L_recon` | 1.0102 @ 0 | 0.3773 @ 14950 | -0.6329 | 0.3655 / 0.394 / 1.0102 | 0.3655 / 0.3753 / 0.3823 (n=100) |
| `L_recon_pred` | 1.0093 @ 0 | 0.383 @ 14950 | -0.6262 | 0.3744 / 0.4002 / 1.0093 | 0.3744 / 0.3836 / 0.3932 (n=100) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 1.0135 @ 14500 | 0.5182 | 0.4952 / 0.9732 / 1.0137 | 1.0097 / 1.0124 / 1.0137 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.1469 @ 14500 | -0.5769 | 0.0978 / 0.1669 / 0.7239 | 0.146 / 0.1484 / 0.1553 (n=10) |
| `c_effective_rank` | 9.4729 @ 0 | 60.3506 @ 14500 | 50.8777 | 9.4729 / 41.3479 / 60.3506 | 57.4486 / 59.4622 / 60.3506 (n=10) |
| `c_plus_effective_rank` | 9.2304 @ 0 | 60.6463 @ 14500 | 51.4159 | 9.2304 / 40.7631 / 60.6463 | 57.2498 / 59.4812 / 60.6463 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 6.677 @ 14500 | -10.0341 | 3.8527 / 7.8648 / 20.337 | 6.4263 / 6.6073 / 6.6804 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.5581 @ 14500 | -0.4418 | 0.5581 / 0.7248 / 1 | 0.5581 / 0.5644 / 0.5769 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 1.321e-08 @ 14500 | -0.9998 | 5.287e-09 / 0.2528 / 0.9999 | 8.356e-09 / 1.162e-08 / 1.370e-08 (n=10) |
| `coarse_copy_loss` | 0.0533 @ 0 | 1.6327 @ 14500 | 1.5794 | 0.0533 / 1.1043 / 1.6327 | 1.5312 / 1.5979 / 1.6327 (n=10) |
| `coarse_model_loss` | 1.2232 @ 0 | 1.7266 @ 14500 | 0.5033 | 0.4193 / 1.3 / 1.7959 | 1.6421 / 1.7107 / 1.7959 (n=10) |
| `coarse_batch_mean_loss` | 0.0502 @ 0 | 1.5063 @ 14500 | 1.4561 | 0.0502 / 1.0146 / 1.5063 | 1.4129 / 1.4743 / 1.5063 (n=10) |
| `coarse_vs_copy_ratio` | 22.9366 @ 0 | 1.0575 @ 14500 | -21.8791 | 0.9312 / 2.1125 / 22.9366 | 1.0179 / 1.0706 / 1.1064 (n=10) |
| `coarse_vs_batch_mean_ratio` | 24.3867 @ 0 | 1.1463 @ 14500 | -23.2404 | 1.0138 / 2.2684 / 24.3867 | 1.1019 / 1.1604 / 1.1987 (n=10) |
| `L_recon_present` | 1.0116 @ 0 | 0.3459 @ 14500 | -0.6658 | 0.3458 / 0.3787 / 1.0116 | 0.3458 / 0.3464 / 0.3476 (n=10) |
| `L_recon_cplus` | 1.0094 @ 0 | 0.3438 @ 14500 | -0.6656 | 0.3438 / 0.3775 / 1.0094 | 0.3438 / 0.3443 / 0.3455 (n=10) |
| `L_recon_chat` | 1.0081 @ 0 | 0.3533 @ 14500 | -0.6548 | 0.3502 / 0.3833 / 1.0081 | 0.3516 / 0.3535 / 0.3552 (n=10) |
| `grad_norm` | 1.5719 @ 0 | 1.8967 @ 14950 | 0.3248 | 0.5025 / 2.1665 / 2.8414 | 1.7514 / 2.4513 / 2.7844 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=60.3506 @ 14500; `c_cross_video_cosine`=0.1469 @ 14500; `c_std_mean`=1.0135 @ 14500; `coarse_vs_copy_ratio`=1.0575 @ 14500; `coarse_vs_batch_mean_ratio`=1.1463 @ 14500; `L_recon_present`=0.3459 @ 14500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.0575 and the latest batch-mean ratio is 1.1463; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 60.3506, cross-video cosine is 0.1469, and copy loss is 1.6327. The reading-cycle verdict is **Healthy rep, no predictor** because Representation health is not the limiting issue; F_c still does not beat copy. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
