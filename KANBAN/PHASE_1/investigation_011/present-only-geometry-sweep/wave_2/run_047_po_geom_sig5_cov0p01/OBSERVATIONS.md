# Observations - run 047 `po_geom_sig5_cov0p01`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `az60m6mx`  
**State:** `crashed`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=crashed; step=14150; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1352 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9833; dead_dim=0; cross_video_cosine=0.0767 |
| Q3 | Rich latent? | PASS | c_effective_rank=150.836; slot_rank=17.8966; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.642 (0.9858 @ 0 -> 0.3438 @ 14000); last=0.3438; train L_recon=0.3722 @ 14150 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.222 @ 0 | 0.0369 @ 14150 | -0.1851 | 0.036 / 0.0488 / 0.222 | 0.036 / 0.0375 / 0.0389 (n=84) |
| `L_flow` | 0 @ 0 | 0 @ 14150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=84) |
| `L_recon` | 0.9829 @ 0 | 0.3722 @ 14150 | -0.6107 | 0.3646 / 0.3976 / 0.9829 | 0.3646 / 0.3738 / 0.3818 (n=84) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=84) |
| `L_recon_present` | 0.9858 @ 0 | 0.3438 @ 14000 | -0.642 | 0.3438 / 0.3812 / 0.9858 | 0.3438 / 0.3448 / 0.3461 (n=9) |
| `recon_scale` | 0 @ 0 | 1 @ 14150 | 1 | 0 / 0.9278 / 1 | 1 / 1 / 1 (n=84) |
| `present_recon_only` | 1 @ 0 | 1 @ 14150 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=84) |
| `prediction_active` | 0 @ 0 | 0 @ 14150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=84) |
| `L_var` | 0.4003 @ 0 | 0.0155 @ 14150 | -0.3848 | 0.0155 / 0.0242 / 0.4003 | 0.0155 / 0.0168 / 0.0185 (n=84) |
| `L_cov` | 2.184 @ 0 | 0.3759 @ 14150 | -1.8081 | 0.363 / 1.1926 / 7.9152 | 0.363 / 0.3896 / 0.428 (n=84) |
| `L_sigreg` | 0.0417 @ 0 | 0.0014 @ 14150 | -0.0404 | 0.0011 / 0.0019 / 0.0417 | 0.0011 / 0.0013 / 0.0015 (n=84) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14150 | 1 | 0 / 0.9278 / 1 | 1 / 1 / 1 (n=84) |
| `c_std_mean` | 0.4952 @ 0 | 0.9833 @ 14000 | 0.4881 | 0.4952 / 0.9538 / 0.9838 | 0.98 / 0.9819 / 0.9838 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0767 @ 14000 | -0.6472 | 0.0671 / 0.1041 / 0.7239 | 0.0759 / 0.0792 / 0.0827 (n=9) |
| `c_effective_rank` | 9.4732 @ 0 | 150.836 @ 14000 | 141.3627 | 9.4732 / 114.8874 / 150.836 | 146.0708 / 149.0534 / 150.836 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 17.8966 @ 14000 | 1.1855 | 3.9128 / 14.909 / 17.8966 | 17.4137 / 17.7118 / 17.8966 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.734 @ 14000 | -0.266 | 0.7338 / 0.7729 / 0.9999 | 0.7338 / 0.7345 / 0.736 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 8.127e-08 @ 14000 | -0.9998 | 4.082e-09 / 0.1144 / 0.9998 | 6.116e-09 / 3.230e-08 / 8.127e-08 (n=9) |
| `grad_norm` | 1.1183 @ 0 | 0.1352 @ 14150 | -0.9831 | 0.1208 / 0.2116 / 1.1183 | 0.1208 / 0.1325 / 0.1513 (n=84) |
| `grad_skipped` | 0 @ 0 | 0 @ 14150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=84) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0097 @ 14150 | 0.0091 | 6.667e-04 / 0.5281 / 1 | 0.0097 / 0.1267 / 0.302 (n=84) |

## Last Key Metrics

`c_effective_rank`=150.836 @ 14000; `c_cross_video_cosine`=0.0767 @ 14000; `c_std_mean`=0.9833 @ 14000; `L_recon_present`=0.3438 @ 14000

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3438, and the key geometry metrics are rank 150.836, cross-video cosine 0.0767, and std 0.9833. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (grid point sig=5.0, cov=0.01):** reached `c_effective_rank` ~150.8 with
cross-video cosine ~0.06-0.10, std ~0.96-0.98, and `L_recon_present` ~0.345 — a decodable, spread,
video-specific present code. Read against the rest of the sweep, the dominant lever on rank is the
COVARIANCE penalty: the cov=0.01 rows reach 139-151, cov=0.003 rows 90-106, and cov=0 rows only
57-83; SIGReg alone (cov=0) tops out ~83, and raising SIGReg helps only modestly on top of cov.
This is a strong PRESENT representation, but F_c is off, so it is not a prediction result.
