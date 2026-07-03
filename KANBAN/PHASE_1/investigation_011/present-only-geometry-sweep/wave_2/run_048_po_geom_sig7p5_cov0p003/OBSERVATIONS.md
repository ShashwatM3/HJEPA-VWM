# Observations - run 048 `po_geom_sig7p5_cov0p003`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `bg7ennr5`  
**State:** `crashed`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=crashed; step=14000; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1814 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9819; dead_dim=0; cross_video_cosine=0.0599 |
| Q3 | Rich latent? | PASS | c_effective_rank=89.8678; slot_rank=9.588; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6387 (0.9858 @ 0 -> 0.3471 @ 14000); last=0.3471; train L_recon=0.3779 @ 14000 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2067 @ 0 | 0.0435 @ 14000 | -0.1632 | 0.0396 / 0.0536 / 0.2067 | 0.0396 / 0.0435 / 0.0465 (n=81) |
| `L_flow` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=81) |
| `L_recon` | 0.9829 @ 0 | 0.3779 @ 14000 | -0.605 | 0.3674 / 0.4036 / 0.9829 | 0.3674 / 0.377 / 0.3846 (n=81) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=81) |
| `L_recon_present` | 0.9858 @ 0 | 0.3471 @ 14000 | -0.6387 | 0.3471 / 0.3865 / 0.9858 | 0.3471 / 0.3479 / 0.3494 (n=9) |
| `recon_scale` | 0 @ 0 | 1 @ 14000 | 1 | 0 / 0.927 / 1 | 1 / 1 / 1 (n=81) |
| `present_recon_only` | 1 @ 0 | 1 @ 14000 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=81) |
| `prediction_active` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=81) |
| `L_var` | 0.4003 @ 0 | 0.0212 @ 14000 | -0.3791 | 0.019 / 0.0273 / 0.4003 | 0.019 / 0.0209 / 0.0233 (n=81) |
| `L_cov` | 2.184 @ 0 | 1.2502 @ 14000 | -0.9338 | 1.1847 / 2.6923 / 10.8681 | 1.1847 / 1.2518 / 1.3354 (n=81) |
| `L_sigreg` | 0.0417 @ 0 | 0.0014 @ 14000 | -0.0404 | 0.001 / 0.0024 / 0.0417 | 0.001 / 0.0014 / 0.0018 (n=81) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14000 | 1 | 0 / 0.927 / 1 | 1 / 1 / 1 (n=81) |
| `c_std_mean` | 0.4952 @ 0 | 0.9819 @ 14000 | 0.4866 | 0.4952 / 0.9491 / 0.9819 | 0.979 / 0.9808 / 0.9819 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0599 @ 14000 | -0.664 | 0.0582 / 0.0996 / 0.7239 | 0.0582 / 0.0613 / 0.0645 (n=9) |
| `c_effective_rank` | 9.4729 @ 0 | 89.8678 @ 14000 | 80.3949 | 9.4729 / 65.5015 / 89.9544 | 86.9229 / 88.6362 / 89.9544 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 9.588 @ 14000 | -7.1231 | 4.0959 / 9.5806 / 19.9247 | 9.1915 / 9.4282 / 9.5884 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6967 @ 14000 | -0.3032 | 0.6914 / 0.7613 / 1 | 0.6952 / 0.6976 / 0.7008 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 6.957e-06 @ 14000 | -0.9998 | 4.865e-09 / 0.1975 / 0.9999 | 1.101e-06 / 4.243e-06 / 8.646e-06 (n=9) |
| `grad_norm` | 1.4368 @ 0 | 0.1814 @ 14000 | -1.2553 | 0.1549 / 0.2992 / 1.5239 | 0.1549 / 0.1833 / 0.2226 (n=81) |
| `grad_skipped` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=81) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0135 @ 14000 | 0.0128 | 6.667e-04 / 0.5336 / 1 | 0.0135 / 0.131 / 0.302 (n=81) |

## Last Key Metrics

`c_effective_rank`=89.8678 @ 14000; `c_cross_video_cosine`=0.0599 @ 14000; `c_std_mean`=0.9819 @ 14000; `L_recon_present`=0.3471 @ 14000

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3471, and the key geometry metrics are rank 89.8678, cross-video cosine 0.0599, and std 0.9819. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
