# Observations - run 030 `lambda_sigreg_3.0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `9jxc8i1q`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=13050; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.7082 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9581; dead_dim=0; cross_video_cosine=0.3516 |
| Q3 | Rich latent? | FAIL | c_effective_rank=34.4457; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1082 (0.0533 @ 0 -> 0.1615 @ 13000); ratio=5.2415; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.8465; copy_loss=0.1615; ratio=5.2415; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=0.884; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | CHECK | present=0.5747; cplus=0.5739; chat=0.6097; chat-cplus=0.0358; reconstruction is active; read with cplus/chat gap and prediction gates |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.2058 @ 0 | 0.8164 @ 13050 | -2.3894 | 0.7379 / 0.9786 / 3.6113 | 0.7809 / 0.8456 / 0.9136 (n=62) |
| `L_flow` | 2.8686 @ 0 | 0.7654 @ 13050 | -2.1032 | 0.6814 / 0.9209 / 3.382 | 0.7317 / 0.7933 / 0.8602 (n=62) |
| `L_var` | 0.4081 @ 0 | 0.0197 @ 13050 | -0.3884 | 0.0134 / 0.0255 / 0.4081 | 0.0152 / 0.0197 / 0.0238 (n=62) |
| `L_sigreg` | 0.0444 @ 0 | 0.0035 @ 13050 | -0.0408 | 0.0028 / 0.0054 / 0.0444 | 0.0028 / 0.0039 / 0.0047 (n=62) |
| `L_recon` | 1.0379 @ 0 | 0.6119 @ 13050 | -0.426 | 0.6041 / 0.6404 / 1.0379 | 0.6065 / 0.6143 / 0.624 (n=62) |
| `L_recon_pred` | 0 @ 0 | 0 @ 13050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=62) |
| `recon_scale` | 0 @ 0 | 1 @ 13050 | 1 | 0 / 0.9218 / 1 | 1 / 1 / 1 (n=62) |
| `c_std_mean` | 0.4952 @ 0 | 0.9581 @ 13000 | 0.4628 | 0.4952 / 0.9414 / 0.9857 | 0.9581 / 0.965 / 0.9716 (n=7) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.3516 @ 13000 | -0.3723 | 0.1206 / 0.2795 / 0.7239 | 0.3245 / 0.3363 / 0.3516 (n=7) |
| `c_effective_rank` | 9.4729 @ 0 | 34.4457 @ 13000 | 24.9728 | 9.4729 / 20.7469 / 34.4457 | 30.0161 / 32.6583 / 34.4457 (n=7) |
| `c_slot_diversity_rank` | 16.711 @ 0 | 2.9519 @ 13000 | -13.7592 | 1.1489 / 7.1617 / 19.9149 | 2.0924 / 2.5655 / 2.9519 (n=7) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.8652 @ 13000 | -0.1348 | 0.8453 / 0.9138 / 1 | 0.8608 / 0.8639 / 0.8652 (n=7) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.2812 @ 13000 | -0.7186 | 0.2724 / 0.581 / 0.9999 | 0.2724 / 0.2764 / 0.2812 (n=7) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1615 @ 13000 | 0.1082 | 0.0533 / 0.2406 / 0.439 | 0.1615 / 0.1737 / 0.1827 (n=7) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.8465 @ 13000 | -2.4641 | 0.6784 / 0.9396 / 3.3106 | 0.6981 / 0.7623 / 0.8465 (n=7) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.9575 @ 13000 | 0.6948 | 0.2627 / 0.9401 / 1.0348 | 0.9575 / 0.975 / 0.9886 (n=7) |
| `coarse_vs_copy_ratio` | 62.0674 @ 0 | 5.2415 @ 13000 | -56.826 | 2.2728 / 5.7006 / 62.0674 | 3.8755 / 4.4026 / 5.2415 (n=7) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.884 @ 13000 | -11.717 | 0.6742 / 1.328 / 12.6011 | 0.7083 / 0.7822 / 0.884 (n=7) |
| `L_recon_present` | 1.0398 @ 0 | 0.5747 @ 13000 | -0.4651 | 0.5747 / 0.6088 / 1.0398 | 0.5747 / 0.5752 / 0.5761 (n=7) |
| `L_recon_cplus` | 1.0407 @ 0 | 0.5739 @ 13000 | -0.4668 | 0.5738 / 0.6111 / 1.0407 | 0.5738 / 0.5752 / 0.577 (n=7) |
| `L_recon_chat` | 1.04 @ 0 | 0.6097 @ 13000 | -0.4303 | 0.5916 / 0.6254 / 1.04 | 0.5942 / 0.5989 / 0.6097 (n=7) |
| `grad_norm` | 2.5654 @ 0 | 2.7082 @ 13050 | 0.1428 | 0.6283 / 2.0512 / 3.0792 | 2.4223 / 2.7025 / 3.0792 (n=62) |
| `grad_skipped` | 0 @ 0 | 0 @ 13050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=62) |
| `grad_has_nan` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0506 @ 13050 | 0.0499 | 6.667e-04 / 0.5702 / 1 | 0.0506 / 0.1622 / 0.302 (n=62) |

## Last Key Metrics

`c_effective_rank`=34.4457 @ 13000; `c_cross_video_cosine`=0.3516 @ 13000; `c_std_mean`=0.9581 @ 13000; `coarse_vs_copy_ratio`=5.2415 @ 13000; `coarse_vs_batch_mean_ratio`=0.884 @ 13000; `L_recon_present`=0.5747 @ 13000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 5.2415 and the latest batch-mean ratio is 0.884; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 34.4457, cross-video cosine is 0.3516, and copy loss is 0.1615. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

It contributed to the SIGReg conclusion: rank can be moved by isotropy pressure, but that movement did not produce a copy-gate win.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
