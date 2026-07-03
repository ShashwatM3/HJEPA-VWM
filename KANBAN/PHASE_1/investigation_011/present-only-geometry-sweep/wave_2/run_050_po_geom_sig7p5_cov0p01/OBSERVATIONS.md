# Observations - run 050 `po_geom_sig7p5_cov0p01`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `h5t89ezx`  
**State:** `crashed`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=crashed; step=14300; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1457 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9696; dead_dim=0; cross_video_cosine=0.0735 |
| Q3 | Rich latent? | PASS | c_effective_rank=146.0266; slot_rank=16.3404; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6403 (0.9858 @ 0 -> 0.3455 @ 14000); last=0.3455; train L_recon=0.3787 @ 14300 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.222 @ 0 | 0.0403 @ 14300 | -0.1817 | 0.0386 / 0.0521 / 0.222 | 0.0386 / 0.0408 / 0.0432 (n=87) |
| `L_flow` | 0 @ 0 | 0 @ 14300 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=87) |
| `L_recon` | 0.9829 @ 0 | 0.3787 @ 14300 | -0.6042 | 0.3651 / 0.3981 / 0.9829 | 0.3651 / 0.375 / 0.383 (n=87) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14300 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=87) |
| `L_recon_present` | 0.9858 @ 0 | 0.3455 @ 14000 | -0.6403 | 0.3455 / 0.3817 / 0.9858 | 0.3455 / 0.3463 / 0.3473 (n=9) |
| `recon_scale` | 0 @ 0 | 1 @ 14300 | 1 | 0 / 0.9286 / 1 | 1 / 1 / 1 (n=87) |
| `present_recon_only` | 1 @ 0 | 1 @ 14300 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=87) |
| `prediction_active` | 0 @ 0 | 0 @ 14300 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=87) |
| `L_var` | 0.4003 @ 0 | 0.0204 @ 14300 | -0.38 | 0.0191 / 0.0276 / 0.4003 | 0.0191 / 0.0208 / 0.0236 (n=87) |
| `L_cov` | 2.184 @ 0 | 0.3767 @ 14300 | -1.8073 | 0.3705 / 1.1793 / 7.9432 | 0.3705 / 0.3991 / 0.4542 (n=87) |
| `L_sigreg` | 0.0417 @ 0 | 9.870e-04 @ 14300 | -0.0407 | 8.327e-04 / 0.0017 / 0.0417 | 8.327e-04 / 0.001 / 0.0012 (n=87) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14300 | 1 | 0 / 0.9286 / 1 | 1 / 1 / 1 (n=87) |
| `c_std_mean` | 0.4952 @ 0 | 0.9696 @ 14000 | 0.4743 | 0.4952 / 0.9434 / 0.973 | 0.966 / 0.9683 / 0.971 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0735 @ 14000 | -0.6504 | 0.057 / 0.0997 / 0.7239 | 0.0708 / 0.0752 / 0.0801 (n=9) |
| `c_effective_rank` | 9.4732 @ 0 | 146.0266 @ 14000 | 136.5533 | 9.4732 / 113.1747 / 146.0266 | 139.6184 / 144.3448 / 146.0266 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 16.3404 @ 14000 | -0.3707 | 4.7229 / 14.0569 / 16.7111 | 16.069 / 16.2439 / 16.3404 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6621 @ 14000 | -0.3379 | 0.6509 / 0.71 / 0.9999 | 0.6609 / 0.6625 / 0.6652 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 2.320e-07 @ 14000 | -0.9998 | 4.111e-09 / 0.1189 / 0.9998 | 1.856e-07 / 1.943e-06 / 1.306e-05 (n=9) |
| `grad_norm` | 1.1183 @ 0 | 0.1457 @ 14300 | -0.9726 | 0.1382 / 0.2402 / 1.1183 | 0.1384 / 0.1556 / 0.181 (n=87) |
| `grad_skipped` | 0 @ 0 | 0 @ 14300 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=87) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0066 @ 14300 | 0.006 | 6.667e-04 / 0.5226 / 1 | 0.0066 / 0.1226 / 0.302 (n=87) |

## Last Key Metrics

`c_effective_rank`=146.0266 @ 14000; `c_cross_video_cosine`=0.0735 @ 14000; `c_std_mean`=0.9696 @ 14000; `L_recon_present`=0.3455 @ 14000

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3455, and the key geometry metrics are rank 146.0266, cross-video cosine 0.0735, and std 0.9696. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
