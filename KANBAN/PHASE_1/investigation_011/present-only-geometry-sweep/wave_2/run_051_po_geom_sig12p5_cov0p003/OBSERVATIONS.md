# Observations - run 051 `po_geom_sig12p5_cov0p003`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `mtrviiab`  
**State:** `crashed`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=crashed; step=14250; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.2103 |
| Q2 | c_t alive / video-specific? | PASS | std=0.958; dead_dim=0; cross_video_cosine=0.0868 |
| Q3 | Rich latent? | PASS | c_effective_rank=106.3076; slot_rank=11.6779; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.636 (0.9858 @ 0 -> 0.3498 @ 14000); last=0.3498; train L_recon=0.3778 @ 14250 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2067 @ 0 | 0.0488 @ 14250 | -0.1579 | 0.0437 / 0.0585 / 0.2067 | 0.0437 / 0.0468 / 0.0505 (n=86) |
| `L_flow` | 0 @ 0 | 0 @ 14250 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=86) |
| `L_recon` | 0.9829 @ 0 | 0.3778 @ 14250 | -0.6051 | 0.369 / 0.4035 / 0.9829 | 0.369 / 0.3791 / 0.3868 (n=86) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14250 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=86) |
| `L_recon_present` | 0.9858 @ 0 | 0.3498 @ 14000 | -0.636 | 0.3498 / 0.3867 / 0.9858 | 0.3498 / 0.3506 / 0.3517 (n=9) |
| `recon_scale` | 0 @ 0 | 1 @ 14250 | 1 | 0 / 0.9283 / 1 | 1 / 1 / 1 (n=86) |
| `present_recon_only` | 1 @ 0 | 1 @ 14250 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=86) |
| `prediction_active` | 0 @ 0 | 0 @ 14250 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=86) |
| `L_var` | 0.4003 @ 0 | 0.0272 @ 14250 | -0.3731 | 0.0232 / 0.0319 / 0.4003 | 0.0232 / 0.0256 / 0.0279 (n=86) |
| `L_cov` | 2.184 @ 0 | 0.901 @ 14250 | -1.283 | 0.8396 / 2.2164 / 10.8509 | 0.8396 / 0.889 / 0.9479 (n=86) |
| `L_sigreg` | 0.0417 @ 0 | 0.0011 @ 14250 | -0.0406 | 7.502e-04 / 0.002 / 0.0417 | 7.502e-04 / 9.950e-04 / 0.0013 (n=86) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14250 | 1 | 0 / 0.9283 / 1 | 1 / 1 / 1 (n=86) |
| `c_std_mean` | 0.4952 @ 0 | 0.958 @ 14000 | 0.4627 | 0.4952 / 0.9291 / 0.9623 | 0.9524 / 0.9565 / 0.9601 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0868 @ 14000 | -0.637 | 0.0708 / 0.1245 / 0.7239 | 0.0826 / 0.0896 / 0.0971 (n=9) |
| `c_effective_rank` | 9.4729 @ 0 | 106.3076 @ 14000 | 96.8348 | 9.4729 / 79.8219 / 106.3076 | 104.5521 / 105.5995 / 106.3076 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 11.6779 @ 14000 | -5.0332 | 6.4473 / 11.3857 / 18.9371 | 11.5697 / 11.6558 / 11.7194 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6445 @ 14000 | -0.3554 | 0.6238 / 0.7105 / 1 | 0.6392 / 0.6427 / 0.6448 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 1.464e-06 @ 14000 | -0.9998 | 6.307e-07 / 0.1846 / 0.9998 | 1.183e-06 / 2.127e-06 / 6.083e-06 (n=9) |
| `grad_norm` | 1.4368 @ 0 | 0.2103 @ 14250 | -1.2264 | 0.1804 / 0.363 / 1.5805 | 0.1804 / 0.2115 / 0.2508 (n=86) |
| `grad_skipped` | 0 @ 0 | 0 @ 14250 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=86) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0076 @ 14250 | 0.0069 | 6.667e-04 / 0.5244 / 1 | 0.0076 / 0.1239 / 0.302 (n=86) |

## Last Key Metrics

`c_effective_rank`=106.3076 @ 14000; `c_cross_video_cosine`=0.0868 @ 14000; `c_std_mean`=0.958 @ 14000; `L_recon_present`=0.3498 @ 14000

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3498, and the key geometry metrics are rank 106.3076, cross-video cosine 0.0868, and std 0.958. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (grid point sig=12.5, cov=0.003):** reached `c_effective_rank` ~106.3 with
cross-video cosine ~0.06-0.10, std ~0.96-0.98, and `L_recon_present` ~0.345 — a decodable, spread,
video-specific present code. Read against the rest of the sweep, the dominant lever on rank is the
COVARIANCE penalty: the cov=0.01 rows reach 139-151, cov=0.003 rows 90-106, and cov=0 rows only
57-83; SIGReg alone (cov=0) tops out ~83, and raising SIGReg helps only modestly on top of cov.
This is a strong PRESENT representation, but F_c is off, so it is not a prediction result.
