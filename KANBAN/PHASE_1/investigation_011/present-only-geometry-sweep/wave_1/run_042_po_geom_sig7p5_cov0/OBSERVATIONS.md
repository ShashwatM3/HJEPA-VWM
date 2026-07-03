# Observations - run 042 `po_geom_sig7p5_cov0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `fq0crddc`  
**State:** `finished`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Low-rank decodable**  
**Verdict note:** The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=finished; step=14950; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.2141 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9678; dead_dim=0; cross_video_cosine=0.1044 |
| Q3 | Rich latent? | PARTIAL | c_effective_rank=56.8112; slot_rank=5.6903; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6399 (0.9858 @ 0 -> 0.3459 @ 14500); last=0.3459; train L_recon=0.3719 @ 14950 |
| Q5 | Geometry and content aligned? | PARTIAL | reconstruction improved through a low-rank or slot-redundant code |
| Q6 | Verdict | Low-rank decodable | The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2002 @ 0 | 0.0447 @ 14950 | -0.1555 | 0.0269 / 0.0507 / 0.2002 | 0.0403 / 0.0434 / 0.0474 (n=100) |
| `L_flow` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon` | 0.9829 @ 0 | 0.3719 @ 14950 | -0.611 | 0.3654 / 0.4026 / 0.9829 | 0.3654 / 0.3748 / 0.3827 (n=100) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon_present` | 0.9858 @ 0 | 0.3459 @ 14500 | -0.6399 | 0.3459 / 0.3868 / 0.9858 | 0.3459 / 0.3466 / 0.348 (n=10) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `present_recon_only` | 1 @ 0 | 1 @ 14950 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=100) |
| `prediction_active` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_var` | 0.4003 @ 0 | 0.0206 @ 14950 | -0.3798 | 0.0177 / 0.025 / 0.4003 | 0.0177 / 0.0196 / 0.0215 (n=100) |
| `L_cov` | 2.184 @ 0 | 2.6118 @ 14950 | 0.4278 | 2.184 / 4.6826 / 14.9717 | 2.4709 / 2.646 / 2.8166 (n=100) |
| `L_sigreg` | 0.0417 @ 0 | 0.0021 @ 14950 | -0.0396 | 0.0015 / 0.0033 / 0.0417 | 0.0015 / 0.002 / 0.0025 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.9678 @ 14500 | 0.4725 | 0.4952 / 0.9411 / 0.9746 | 0.9632 / 0.9665 / 0.9681 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.1044 @ 14500 | -0.6195 | 0.0904 / 0.1324 / 0.7239 | 0.1036 / 0.1064 / 0.1122 (n=10) |
| `c_effective_rank` | 9.4728 @ 0 | 56.8112 @ 14500 | 47.3384 | 9.4728 / 40.2324 / 56.8357 | 52.918 / 55.3593 / 56.8357 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 5.6903 @ 14500 | -11.0208 | 3.6884 / 7.25 / 20.7176 | 5.3455 / 5.6108 / 5.7013 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.755 @ 14500 | -0.2449 | 0.7515 / 0.8244 / 1 | 0.755 / 0.7619 / 0.7772 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 7.310e-09 @ 14500 | -0.9998 | 4.083e-09 / 0.2404 / 0.9999 | 4.097e-09 / 5.873e-09 / 8.695e-09 (n=10) |
| `grad_norm` | 1.5798 @ 0 | 0.2141 @ 14950 | -1.3656 | 0.1875 / 0.3257 / 1.7475 | 0.1875 / 0.215 / 0.2662 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=56.8112 @ 14500; `c_cross_video_cosine`=0.1044 @ 14500; `c_std_mean`=0.9678 @ 14500; `L_recon_present`=0.3459 @ 14500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3459, and the key geometry metrics are rank 56.8112, cross-video cosine 0.1044, and std 0.9678. The reading-cycle verdict is **Low-rank decodable** because The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
