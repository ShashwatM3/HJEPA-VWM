# Observations - run 052 `ae_sharp_slots_recon_only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `662hfy3c`  
**State:** `running`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Collapsed rep**  
**Verdict note:** Present reconstruction may improve, but c_t is still weakly spread or video-independent.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=running; step=5600; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.0111 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4085; dead_dim=0; cross_video_cosine=0.8161 |
| Q3 | Rich latent? | FAIL | c_effective_rank=22.0355; slot_rank=12.8173; centered_slot_rank=12.1236 |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.7015 (1.0127 @ 0 -> 0.3113 @ 5500); last=0.3113; train L_recon=0.3444 @ 5600 |
| Q5 | Geometry and content aligned? | FAIL | reconstruction improved, but c_t remains collapsed/video-independent or weakly spread |
| Q6 | Verdict | Collapsed rep | Present reconstruction may improve, but c_t is still weakly spread or video-independent. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0 @ 0 | 0.0172 @ 5600 | 0.0172 | 0 / 0.0149 / 0.019 | n/a |
| `L_flow` | 0 @ 0 | 0 @ 5600 | 0 | 0 / 0 / 0 | n/a |
| `L_recon` | 1.0025 @ 0 | 0.3444 @ 5600 | -0.6582 | 0.3338 / 0.3867 / 1.0025 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 5600 | 0 | 0 / 0 / 0 | n/a |
| `L_recon_present` | 1.0127 @ 0 | 0.3113 @ 5500 | -0.7015 | 0.3113 / 0.3956 / 1.0127 | n/a |
| `recon_scale` | 0 @ 0 | 1 @ 5600 | 1 | 0 / 0.8186 / 1 | n/a |
| `present_recon_only` | 1 @ 0 | 1 @ 5600 | 0 | 1 / 1 / 1 | n/a |
| `prediction_active` | 0 @ 0 | 0 @ 5600 | 0 | 0 / 0 / 0 | n/a |
| `L_var` | 1 @ 0 | 0.5558 @ 5600 | -0.4442 | 0.5524 / 0.6948 / 1 | n/a |
| `L_cov` | 6.7258 @ 0 | 19.0816 @ 5600 | 12.3558 | 6.5244 / 13.6104 / 19.287 | n/a |
| `L_sigreg` | 0.0073 @ 0 | 0.0093 @ 5600 | 0.002 | 0.0059 / 0.0081 / 0.0111 | n/a |
| `sigreg_scale` | 0 @ 0 | 0 @ 5600 | 0 | 0 / 0 / 0 | n/a |
| `c_std_mean` | 0 @ 0 | 0.4085 @ 5500 | 0.4085 | 0 / 0.2735 / 0.4114 | n/a |
| `c_dead_dim_frac` | 1 @ 0 | 0 @ 5500 | -1 | 0 / 0.0833 / 1 | n/a |
| `c_cross_video_cosine` | 1 @ 0 | 0.8161 @ 5500 | -0.1839 | 0.8129 / 0.8921 / 1 | n/a |
| `c_effective_rank` | 30.9931 @ 0 | 22.0355 @ 5500 | -8.9576 | 22.0355 / 26.7259 / 32.0687 | n/a |
| `c_slot_diversity_rank` | 31.993 @ 0 | 12.8173 @ 5500 | -19.1757 | 12.8173 / 20.8363 / 31.993 | n/a |
| `c_slot_diversity_rank_centered` | 30.993 @ 0 | 12.1236 @ 5500 | -18.8694 | 12.1236 / 20.1022 / 30.993 | n/a |
| `c_attn_entropy` | 0.7551 @ 0 | 0.57 @ 5500 | -0.185 | 0.57 / 0.6594 / 0.7564 | n/a |
| `c_attn_entropy_min` | 0.3284 @ 0 | 0.3926 @ 5500 | 0.0642 | 0.3284 / 0.4191 / 0.4591 | n/a |
| `grad_norm` | 0 @ 0 | 0.0111 @ 5600 | 0.0111 | 0 / 0.0139 / 0.041 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 5600 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 5500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.7892 @ 5600 | 0.7885 | 6.667e-04 / 0.8095 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=22.0355 @ 5500; `c_cross_video_cosine`=0.8161 @ 5500; `c_std_mean`=0.4085 @ 5500; `L_recon_present`=0.3113 @ 5500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3113, and the key geometry metrics are rank 22.0355, cross-video cosine 0.8161, and std 0.4085. The reading-cycle verdict is **Collapsed rep** because Present reconstruction may improve, but c_t is still weakly spread or video-independent. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It is testing whether bottleneck architecture alone can replace explicit geometry regularizers. Current live evidence says reconstruction improves before geometry becomes healthy.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
