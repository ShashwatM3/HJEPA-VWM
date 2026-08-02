# Observations - run 043 `po_geom_sig5_cov0p003`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `4f2p1e7b`  
**State:** `finished`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=finished; step=14950; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1602 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9847; dead_dim=0; cross_video_cosine=0.0738 |
| Q3 | Rich latent? | PASS | c_effective_rank=91.0551; slot_rank=9.9985; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6387 (0.9858 @ 0 -> 0.3471 @ 14500); last=0.3471; train L_recon=0.3721 @ 14950 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2067 @ 0 | 0.0388 @ 14950 | -0.1679 | 0.0366 / 0.0487 / 0.2067 | 0.0366 / 0.0388 / 0.0409 (n=100) |
| `L_flow` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon` | 0.9829 @ 0 | 0.3721 @ 14950 | -0.6108 | 0.3661 / 0.4025 / 0.9829 | 0.3661 / 0.3751 / 0.3828 (n=100) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon_present` | 0.9858 @ 0 | 0.3471 @ 14500 | -0.6387 | 0.3471 / 0.3868 / 0.9858 | 0.3471 / 0.3478 / 0.349 (n=10) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `present_recon_only` | 1 @ 0 | 1 @ 14950 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=100) |
| `prediction_active` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_var` | 0.4003 @ 0 | 0.0173 @ 14950 | -0.383 | 0.0151 / 0.0234 / 0.4003 | 0.0151 / 0.0169 / 0.0193 (n=100) |
| `L_cov` | 2.184 @ 0 | 1.1625 @ 14950 | -1.0215 | 1.1287 / 2.7601 / 10.8758 | 1.1287 / 1.1948 / 1.3421 (n=100) |
| `L_sigreg` | 0.0417 @ 0 | 0.0016 @ 14950 | -0.0401 | 0.0012 / 0.0026 / 0.0417 | 0.0012 / 0.0016 / 0.0019 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.9847 @ 14500 | 0.4895 | 0.4952 / 0.9548 / 0.9857 | 0.9816 / 0.9837 / 0.9857 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0738 @ 14500 | -0.65 | 0.0564 / 0.1037 / 0.7239 | 0.0713 / 0.0754 / 0.0798 (n=10) |
| `c_effective_rank` | 9.4729 @ 0 | 91.0551 @ 14500 | 81.5823 | 9.4729 / 63.664 / 91.0551 | 86.834 / 89.5527 / 91.0551 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 9.9985 @ 14500 | -6.7126 | 4.2813 / 9.8667 / 20.3502 | 9.4857 / 9.8468 / 9.9985 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.7062 @ 14500 | -0.2937 | 0.7012 / 0.7842 / 1 | 0.7012 / 0.7057 / 0.7073 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 4.388e-09 @ 14500 | -0.9998 | 4.187e-09 / 0.2176 / 0.9999 | 4.343e-09 / 4.894e-09 / 7.278e-09 (n=10) |
| `grad_norm` | 1.4368 @ 0 | 0.1602 @ 14950 | -1.2766 | 0.1208 / 0.2442 / 1.4957 | 0.1208 / 0.1451 / 0.1633 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=91.0551 @ 14500; `c_cross_video_cosine`=0.0738 @ 14500; `c_std_mean`=0.9847 @ 14500; `L_recon_present`=0.3471 @ 14500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3471, and the key geometry metrics are rank 91.0551, cross-video cosine 0.0738, and std 0.9847. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (grid point sig=5.0, cov=0.003):** reached `c_effective_rank` ~91.1 with
cross-video cosine ~0.06-0.10, std ~0.96-0.98, and `L_recon_present` ~0.345 — a decodable, spread,
video-specific present code. Read against the rest of the sweep, the dominant lever on rank is the
COVARIANCE penalty: the cov=0.01 rows reach 139-151, cov=0.003 rows 90-106, and cov=0 rows only
57-83; SIGReg alone (cov=0) tops out ~83, and raising SIGReg helps only modestly on top of cov.
This is a strong PRESENT representation, but F_c is off, so it is not a prediction result.
