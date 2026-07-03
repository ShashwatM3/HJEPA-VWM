# Observations - run 045 `po_geom_sig12p5_cov0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `76d6o8d2`  
**State:** `finished`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=finished; step=14950; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.2511 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9628; dead_dim=0; cross_video_cosine=0.0991 |
| Q3 | Rich latent? | PASS | c_effective_rank=83.4092; slot_rank=9.0834; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.634 (0.9858 @ 0 -> 0.3518 @ 14500); last=0.3518; train L_recon=0.3778 @ 14950 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2002 @ 0 | 0.0486 @ 14950 | -0.1515 | 0.0296 / 0.0565 / 0.2002 | 0.044 / 0.0471 / 0.051 (n=100) |
| `L_flow` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon` | 0.9829 @ 0 | 0.3778 @ 14950 | -0.6051 | 0.3706 / 0.4035 / 0.9829 | 0.3706 / 0.3802 / 0.3881 (n=100) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon_present` | 0.9858 @ 0 | 0.3518 @ 14500 | -0.634 | 0.3518 / 0.3873 / 0.9858 | 0.3518 / 0.3526 / 0.3539 (n=10) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `present_recon_only` | 1 @ 0 | 1 @ 14950 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=100) |
| `prediction_active` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_var` | 0.4003 @ 0 | 0.0257 @ 14950 | -0.3746 | 0.0224 / 0.0308 / 0.4003 | 0.0224 / 0.025 / 0.0271 (n=100) |
| `L_cov` | 2.184 @ 0 | 1.4531 @ 14950 | -0.7309 | 1.4129 / 3.2674 / 14.3756 | 1.4129 / 1.4829 / 1.6105 (n=100) |
| `L_sigreg` | 0.0417 @ 0 | 0.0014 @ 14950 | -0.0404 | 9.493e-04 / 0.0024 / 0.0417 | 9.493e-04 / 0.0012 / 0.0016 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.9628 @ 14500 | 0.4675 | 0.4952 / 0.9325 / 0.9744 | 0.9587 / 0.9624 / 0.9651 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0991 @ 14500 | -0.6247 | 0.0758 / 0.1365 / 0.7239 | 0.0948 / 0.0999 / 0.107 (n=10) |
| `c_effective_rank` | 9.4728 @ 0 | 83.4092 @ 14500 | 73.9364 | 9.4728 / 61.9548 / 83.4562 | 82.0011 / 82.9113 / 83.4562 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 9.0834 @ 14500 | -7.6276 | 4.2014 / 9.4222 / 19.9263 | 8.9113 / 9.0558 / 9.0994 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6081 @ 14500 | -0.3918 | 0.6073 / 0.703 / 1 | 0.6081 / 0.6095 / 0.6117 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 9.834e-07 @ 14500 | -0.9998 | 2.761e-07 / 0.2007 / 0.9999 | 2.761e-07 / 2.668e-06 / 1.428e-05 (n=10) |
| `grad_norm` | 1.5798 @ 0 | 0.2511 @ 14950 | -1.3286 | 0.2049 / 0.3884 / 1.8039 | 0.2049 / 0.2345 / 0.2875 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=83.4092 @ 14500; `c_cross_video_cosine`=0.0991 @ 14500; `c_std_mean`=0.9628 @ 14500; `L_recon_present`=0.3518 @ 14500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3518, and the key geometry metrics are rank 83.4092, cross-video cosine 0.0991, and std 0.9628. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
