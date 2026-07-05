# Observations - run 049 `po_geom_sig10_cov0p01`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `bttexglp`  
**State:** `crashed`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=crashed; step=14350; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1854 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9635; dead_dim=0; cross_video_cosine=0.0742 |
| Q3 | Rich latent? | PASS | c_effective_rank=139.0824; slot_rank=15.7357; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6373 (0.9858 @ 0 -> 0.3485 @ 14000); last=0.3485; train L_recon=0.3745 @ 14350 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.222 @ 0 | 0.0442 @ 14350 | -0.1778 | 0.0427 / 0.056 / 0.222 | 0.0427 / 0.0444 / 0.0471 (n=88) |
| `L_flow` | 0 @ 0 | 0 @ 14350 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=88) |
| `L_recon` | 0.9829 @ 0 | 0.3745 @ 14350 | -0.6084 | 0.3687 / 0.4001 / 0.9829 | 0.3687 / 0.3782 / 0.386 (n=88) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14350 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=88) |
| `L_recon_present` | 0.9858 @ 0 | 0.3485 @ 14000 | -0.6373 | 0.3484 / 0.3837 / 0.9858 | 0.3484 / 0.3494 / 0.3508 (n=9) |
| `recon_scale` | 0 @ 0 | 1 @ 14350 | 1 | 0 / 0.9288 / 1 | 1 / 1 / 1 (n=88) |
| `present_recon_only` | 1 @ 0 | 1 @ 14350 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=88) |
| `prediction_active` | 0 @ 0 | 0 @ 14350 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=88) |
| `L_var` | 0.4003 @ 0 | 0.0234 @ 14350 | -0.3769 | 0.0225 / 0.0304 / 0.4003 | 0.0225 / 0.0239 / 0.026 (n=88) |
| `L_cov` | 2.184 @ 0 | 0.4482 @ 14350 | -1.7358 | 0.4143 / 1.2282 / 7.9695 | 0.4143 / 0.4509 / 0.5024 (n=88) |
| `L_sigreg` | 0.0417 @ 0 | 9.317e-04 @ 14350 | -0.0408 | 7.230e-04 / 0.0016 / 0.0417 | 7.230e-04 / 9.042e-04 / 0.0011 (n=88) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14350 | 1 | 0 / 0.9288 / 1 | 1 / 1 / 1 (n=88) |
| `c_std_mean` | 0.4952 @ 0 | 0.9635 @ 14000 | 0.4682 | 0.4952 / 0.9376 / 0.9664 | 0.9594 / 0.9631 / 0.9663 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0742 @ 14000 | -0.6496 | 0.0627 / 0.103 / 0.7239 | 0.0687 / 0.0747 / 0.0816 (n=9) |
| `c_effective_rank` | 9.4732 @ 0 | 139.0824 @ 14000 | 129.6091 | 9.4732 / 107.6995 / 139.407 | 132.3898 / 136.8533 / 139.407 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 15.7357 @ 14000 | -0.9754 | 5.6304 / 13.4908 / 16.7111 | 15.1657 / 15.5406 / 15.748 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6731 @ 14000 | -0.3269 | 0.6563 / 0.7123 / 0.9999 | 0.6672 / 0.6719 / 0.6742 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 8.188e-06 @ 14000 | -0.9998 | 4.145e-09 / 0.1214 / 0.9998 | 5.506e-09 / 4.170e-06 / 8.678e-06 (n=9) |
| `grad_norm` | 1.1183 @ 0 | 0.1854 @ 14350 | -0.9329 | 0.1532 / 0.2728 / 1.1256 | 0.1532 / 0.1813 / 0.2034 (n=88) |
| `grad_skipped` | 0 @ 0 | 0 @ 14350 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=88) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0057 @ 14350 | 0.005 | 6.667e-04 / 0.5208 / 1 | 0.0057 / 0.1213 / 0.302 (n=88) |

## Last Key Metrics

`c_effective_rank`=139.0824 @ 14000; `c_cross_video_cosine`=0.0742 @ 14000; `c_std_mean`=0.9635 @ 14000; `L_recon_present`=0.3485 @ 14000

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3485, and the key geometry metrics are rank 139.0824, cross-video cosine 0.0742, and std 0.9635. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (grid point sig=10.0, cov=0.01):** reached `c_effective_rank` ~139.1 with
cross-video cosine ~0.06-0.10, std ~0.96-0.98, and `L_recon_present` ~0.345 — a decodable, spread,
video-specific present code. Read against the rest of the sweep, the dominant lever on rank is the
COVARIANCE penalty: the cov=0.01 rows reach 139-151, cov=0.003 rows 90-106, and cov=0 rows only
57-83; SIGReg alone (cov=0) tops out ~83, and raising SIGReg helps only modestly on top of cov.
This is a strong PRESENT representation, but F_c is off, so it is not a prediction result.
