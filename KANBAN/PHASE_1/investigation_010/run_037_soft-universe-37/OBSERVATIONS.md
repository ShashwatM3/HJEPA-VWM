# Observations - run 037 `soft-universe-37`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `2vbo6pbm`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Healthy rep, no predictor**  
**Verdict note:** Representation health is not the limiting issue; F_c still does not beat copy.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | PASS | state=finished; step=14950; grad_skipped max=0; grad_has_nan max=0; grad_norm last=1.8317 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0054; dead_dim=0; cross_video_cosine=0.1624 |
| Q3 | Rich latent? | PASS | c_effective_rank=61.0804; c_plus_effective_rank=60.4079 |
| Q4 | Temporal dynamics? | PARTIAL | copy_loss trend=up 1.5091 (0.0533 @ 0 -> 1.5625 @ 14500); ratio=1.0631; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.6611; copy_loss=1.5625; ratio=1.0631; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.1397; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5713; cplus=0.5683; chat=0.5804; chat-cplus=0.0121; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Healthy rep, no predictor | Representation health is not the limiting issue; F_c still does not beat copy. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 1.3752 @ 0 | 1.7013 @ 14950 | 0.3261 | 0.4423 / 1.3948 / 1.9383 | 1.6687 / 1.8056 / 1.9383 (n=100) |
| `L_flow` | 1.1691 @ 0 | 1.6205 @ 14950 | 0.4514 | 0.3362 / 1.309 / 1.8549 | 1.5867 / 1.7238 / 1.8549 (n=100) |
| `L_var` | 0.4121 @ 0 | 0.0153 @ 14950 | -0.3968 | 0.0142 / 0.0223 / 0.4121 | 0.0142 / 0.0163 / 0.018 (n=100) |
| `L_sigreg` | 0.0444 @ 0 | 0.0023 @ 14950 | -0.0421 | 0.0019 / 0.0039 / 0.0444 | 0.0019 / 0.0024 / 0.0031 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `L_recon` | 1.0389 @ 0 | 0.6122 @ 14950 | -0.4267 | 0.5968 / 0.6332 / 1.0389 | 0.5968 / 0.61 / 0.6189 (n=100) |
| `L_recon_pred` | 1.0385 @ 0 | 0.6194 @ 14950 | -0.4191 | 0.6088 / 0.6409 / 1.0385 | 0.6088 / 0.6206 / 0.6325 (n=100) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 1.0054 @ 14500 | 0.5101 | 0.4952 / 0.9724 / 1.0129 | 1.005 / 1.0064 / 1.009 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.1624 @ 14500 | -0.5615 | 0.1135 / 0.1684 / 0.7239 | 0.1548 / 0.1602 / 0.163 (n=10) |
| `c_effective_rank` | 9.4729 @ 0 | 61.0804 @ 14500 | 51.6075 | 9.4729 / 41.8692 / 61.1009 | 58.6979 / 60.4571 / 61.1009 (n=10) |
| `c_plus_effective_rank` | 9.2304 @ 0 | 60.4079 @ 14500 | 51.1775 | 9.2304 / 40.2482 / 60.4079 | 56.9685 / 59.3428 / 60.4079 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 7.3634 @ 14500 | -9.3476 | 3.8576 / 8.3657 / 20.4728 | 7.0164 / 7.2577 / 7.3648 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.5746 @ 14500 | -0.4254 | 0.5746 / 0.7275 / 1 | 0.5746 / 0.5792 / 0.5887 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 4.109e-09 @ 14500 | -0.9998 | 4.109e-09 / 0.2578 / 0.9999 | 4.109e-09 / 4.239e-09 / 4.707e-09 (n=10) |
| `coarse_copy_loss` | 0.0533 @ 0 | 1.5625 @ 14500 | 1.5091 | 0.0533 / 1.0707 / 1.5625 | 1.4908 / 1.5388 / 1.5625 (n=10) |
| `coarse_model_loss` | 1.2232 @ 0 | 1.6611 @ 14500 | 0.4379 | 0.3808 / 1.2664 / 1.7205 | 1.5973 / 1.6578 / 1.7205 (n=10) |
| `coarse_batch_mean_loss` | 0.0502 @ 0 | 1.4575 @ 14500 | 1.4074 | 0.0502 / 0.9968 / 1.4575 | 1.3885 / 1.4332 / 1.4575 (n=10) |
| `coarse_vs_copy_ratio` | 22.9366 @ 0 | 1.0631 @ 14500 | -21.8735 | 0.9333 / 2.1138 / 22.9366 | 1.036 / 1.0774 / 1.1067 (n=10) |
| `coarse_vs_batch_mean_ratio` | 24.3867 @ 0 | 1.1397 @ 14500 | -23.247 | 1.0196 / 2.2573 / 24.3867 | 1.1126 / 1.1568 / 1.1893 (n=10) |
| `L_recon_present` | 1.0386 @ 0 | 0.5713 @ 14500 | -0.4673 | 0.5713 / 0.6024 / 1.0386 | 0.5713 / 0.5719 / 0.5729 (n=10) |
| `L_recon_cplus` | 1.0378 @ 0 | 0.5683 @ 14500 | -0.4695 | 0.5683 / 0.6009 / 1.0378 | 0.5683 / 0.5689 / 0.5704 (n=10) |
| `L_recon_chat` | 1.0373 @ 0 | 0.5804 @ 14500 | -0.4569 | 0.5766 / 0.6081 / 1.0373 | 0.5775 / 0.5807 / 0.583 (n=10) |
| `grad_norm` | 1.5719 @ 0 | 1.8317 @ 14950 | 0.2598 | 0.5021 / 2.2342 / 2.8892 | 1.8317 / 2.5224 / 2.8115 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=61.0804 @ 14500; `c_cross_video_cosine`=0.1624 @ 14500; `c_std_mean`=1.0054 @ 14500; `coarse_vs_copy_ratio`=1.0631 @ 14500; `coarse_vs_batch_mean_ratio`=1.1397 @ 14500; `L_recon_present`=0.5713 @ 14500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.0631 and the latest batch-mean ratio is 1.1397; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 61.0804, cross-video cosine is 0.1624, and copy loss is 1.5625. The reading-cycle verdict is **Healthy rep, no predictor** because Representation health is not the limiting issue; F_c still does not beat copy. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

It became the canonical negative result: a healthy, high-rank, video-specific representation can still fail to forecast better than copy.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
