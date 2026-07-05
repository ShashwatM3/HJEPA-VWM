# Observations - run 032 `lambda_sigreg_0.3`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `x7z6e0ah`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=13150; grad_skipped max=0; grad_has_nan max=0; grad_norm last=3.0377 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0191; dead_dim=0; cross_video_cosine=0.3067 |
| Q3 | Rich latent? | FAIL | c_effective_rank=13.2162; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1354 (0.0533 @ 0 -> 0.1887 @ 13000); ratio=2.3375; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.4412; copy_loss=0.1887; ratio=2.3375; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.4012; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5878; cplus=0.5857; chat=0.5968; chat-cplus=0.0111; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.086 @ 0 | 0.4931 @ 13150 | -2.5929 | 0.4279 / 0.7976 / 3.5428 | 0.4279 / 0.4948 / 0.5521 (n=64) |
| `L_flow` | 2.8686 @ 0 | 0.4562 @ 13150 | -2.4124 | 0.39 / 0.7582 / 3.3819 | 0.39 / 0.4578 / 0.5151 (n=64) |
| `L_var` | 0.4081 @ 0 | 0.0061 @ 13150 | -0.402 | 0.0038 / 0.0152 / 0.4081 | 0.0039 / 0.0068 / 0.0131 (n=64) |
| `L_sigreg` | 0.0444 @ 0 | 0.0089 @ 13150 | -0.0355 | 0.0063 / 0.0087 / 0.0444 | 0.0064 / 0.008 / 0.0098 (n=64) |
| `L_recon` | 1.0379 @ 0 | 0.6232 @ 13150 | -0.4147 | 0.6106 / 0.6454 / 1.0379 | 0.6175 / 0.6254 / 0.6346 (n=64) |
| `L_recon_pred` | 0 @ 0 | 0 @ 13150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=64) |
| `recon_scale` | 0 @ 0 | 1 @ 13150 | 1 | 0 / 0.9223 / 1 | 1 / 1 / 1 (n=64) |
| `c_std_mean` | 0.4952 @ 0 | 1.0191 @ 13000 | 0.5239 | 0.4952 / 0.9747 / 1.048 | 1.0099 / 1.0271 / 1.048 (n=7) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.3067 @ 13000 | -0.4172 | 0.1136 / 0.2445 / 0.7239 | 0.2514 / 0.2847 / 0.3067 (n=7) |
| `c_effective_rank` | 9.4728 @ 0 | 13.2162 @ 13000 | 3.7434 | 9.4728 / 11.9099 / 13.2669 | 12.8993 / 13.1594 / 13.2669 (n=7) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.1925 @ 13000 | 1.4814 | 16.7111 / 17.9473 / 20.5993 | 18.1171 / 18.2295 / 18.311 (n=7) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.8883 @ 13000 | -0.1116 | 0.8883 / 0.9339 / 1 | 0.8883 / 0.8925 / 0.8988 (n=7) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.7811 @ 13000 | -0.2187 | 0.7809 / 0.8592 / 0.9999 | 0.7809 / 0.7863 / 0.7915 (n=7) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1887 @ 13000 | 0.1354 | 0.0533 / 0.2486 / 0.3827 | 0.1887 / 0.21 / 0.2415 (n=7) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.4412 @ 13000 | -2.8694 | 0.3674 / 0.7856 / 3.3106 | 0.3674 / 0.4485 / 0.5131 (n=7) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.0997 @ 13000 | 0.8369 | 0.2627 / 1.0106 / 1.1519 | 1.0878 / 1.1165 / 1.1519 (n=7) |
| `coarse_vs_copy_ratio` | 62.0671 @ 0 | 2.3375 @ 13000 | -59.7296 | 1.7714 / 4.8458 / 62.0671 | 1.788 / 2.1385 / 2.4005 (n=7) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.4012 @ 13000 | -12.1999 | 0.3378 / 1.1424 / 12.6011 | 0.3378 / 0.4017 / 0.4677 (n=7) |
| `L_recon_present` | 1.0398 @ 0 | 0.5878 @ 13000 | -0.452 | 0.5867 / 0.6148 / 1.0398 | 0.5873 / 0.5879 / 0.5883 (n=7) |
| `L_recon_cplus` | 1.0407 @ 0 | 0.5857 @ 13000 | -0.455 | 0.5849 / 0.6164 / 1.0407 | 0.5849 / 0.5859 / 0.5881 (n=7) |
| `L_recon_chat` | 1.04 @ 0 | 0.5968 @ 13000 | -0.4432 | 0.5909 / 0.6243 / 1.04 | 0.5909 / 0.5932 / 0.5968 (n=7) |
| `grad_norm` | 1.6131 @ 0 | 3.0377 @ 13150 | 1.4246 | 0.5361 / 2.2749 / 3.6887 | 2.7519 / 3.1758 / 3.6887 (n=64) |
| `grad_skipped` | 0 @ 0 | 0 @ 13150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=64) |
| `grad_has_nan` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0456 @ 13150 | 0.045 | 6.667e-04 / 0.5662 / 1 | 0.0456 / 0.1586 / 0.302 (n=64) |

## Last Key Metrics

`c_effective_rank`=13.2162 @ 13000; `c_cross_video_cosine`=0.3067 @ 13000; `c_std_mean`=1.0191 @ 13000; `coarse_vs_copy_ratio`=2.3375 @ 13000; `coarse_vs_batch_mean_ratio`=0.4012 @ 13000; `L_recon_present`=0.5878 @ 13000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.3375 and the latest batch-mean ratio is 0.4012; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 13.2162, cross-video cosine is 0.3067, and copy loss is 0.1887. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

It contributed to the SIGReg conclusion: rank can be moved by isotropy pressure, but that movement did not produce a copy-gate win.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (W&B `x7z6e0ah`):** lambda_sigreg=0.3 is essentially the control — rank ~13.2,
copy ratio ~2.18. It has the BEST copy ratio of the sweep, but only because it barely perturbs
the baseline; it is nowhere near the <0.70 gate. Confirms a light isotropy push does nothing to
rank and cannot help prediction.
