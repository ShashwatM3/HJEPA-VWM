# Observations - run 039 `original_recon_loss + no-pred`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `kttd1fib`  
**State:** `finished`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Collapsed rep**  
**Verdict note:** Present reconstruction may improve, but c_t is still weakly spread or video-independent.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=finished; step=14950; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.0074 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.3472; dead_dim=0; cross_video_cosine=0.8638 |
| Q3 | Rich latent? | FAIL | c_effective_rank=10.464; slot_rank=6.2264; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.5237 (1.0386 @ 0 -> 0.5149 @ 14500); last=0.5149; train L_recon=0.5585 @ 14950 |
| Q5 | Geometry and content aligned? | FAIL | reconstruction improved, but c_t remains collapsed/video-independent or weakly spread |
| Q6 | Verdict | Collapsed rep | Present reconstruction may improve, but c_t is still weakly spread or video-independent. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0 @ 0 | 0.0279 @ 14950 | 0.0279 | 0 / 0.0272 / 0.0317 | 0.0272 / 0.0279 / 0.0283 (n=100) |
| `L_flow` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon` | 1.0389 @ 0 | 0.5585 @ 14950 | -0.4805 | 0.5446 / 0.5981 / 1.0389 | 0.5446 / 0.5572 / 0.567 (n=100) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon_present` | 1.0386 @ 0 | 0.5149 @ 14500 | -0.5237 | 0.5149 / 0.567 / 1.0386 | 0.5149 / 0.5172 / 0.5215 (n=10) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `present_recon_only` | 1 @ 0 | 1 @ 14950 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=100) |
| `prediction_active` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_var` | 0.4121 @ 0 | 0.6175 @ 14950 | 0.2054 | 0.4121 / 0.5528 / 0.6891 | 0.5641 / 0.6044 / 0.6285 (n=100) |
| `L_cov` | 1.687 @ 0 | 28.7831 @ 14950 | 27.0961 | 0.1641 / 13.3332 / 29.2288 | 23.2281 / 27.1481 / 29.2288 (n=100) |
| `L_sigreg` | 0.0444 @ 0 | 0.0168 @ 14950 | -0.0276 | 0.0107 / 0.0363 / 0.1104 | 0.0126 / 0.017 / 0.0229 (n=100) |
| `sigreg_scale` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.3472 @ 14500 | -0.148 | 0.332 / 0.4181 / 0.4952 | 0.3472 / 0.3657 / 0.3972 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.724 @ 0 | 0.8638 @ 14500 | 0.1398 | 0.724 / 0.7998 / 0.875 | 0.8218 / 0.8486 / 0.8638 (n=10) |
| `c_effective_rank` | 9.473 @ 0 | 10.464 @ 14500 | 0.991 | 7.7991 / 13.1833 / 19.9306 | 10.464 / 11.306 / 12.8211 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 6.2264 @ 14500 | -10.4848 | 1.4869 / 7.1969 / 16.7111 | 6.2264 / 6.3808 / 6.6292 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.5867 @ 14500 | -0.4133 | 0.5867 / 0.7997 / 0.9999 | 0.5867 / 0.6031 / 0.6349 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.3641 @ 14500 | -0.6358 | 0.3641 / 0.5794 / 0.9998 | 0.3641 / 0.3718 / 0.3798 (n=10) |
| `grad_norm` | 0 @ 0 | 0.0074 @ 14950 | 0.0074 | 0 / 0.0096 / 0.0208 | 0.0071 / 0.0115 / 0.0156 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=10.464 @ 14500; `c_cross_video_cosine`=0.8638 @ 14500; `c_std_mean`=0.3472 @ 14500; `L_recon_present`=0.5149 @ 14500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.5149, and the key geometry metrics are rank 10.464, cross-video cosine 0.8638, and std 0.3472. The reading-cycle verdict is **Collapsed rep** because Present reconstruction may improve, but c_t is still weakly spread or video-independent. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (W&B `kttd1fib`):** it collapsed. `c_effective_rank` ~10.5, cross-video cosine
~0.86, std ~0.35 — a video-independent, low-rank code — even though present reconstruction still
improved to ~0.515. Verdict Collapsed-rep. This is the precursor to run 052: present-only
reconstruction with NO geometry regularizers produces a template-like code. It establishes that
the strong present representations seen later in this investigation (runs 042-051) come from the
GEOMETRY REGULARIZERS, not from reconstruction alone.
