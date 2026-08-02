# Observations - run 040 `inv011_fixed_position_decoder`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `io74f32b`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | PASS | state=finished; step=14950; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.2759 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0053; dead_dim=0; cross_video_cosine=0.1652 |
| Q3 | Rich latent? | PARTIAL | c_effective_rank=50.7635; c_plus_effective_rank=50.4543 |
| Q4 | Temporal dynamics? | PARTIAL | copy_loss trend=up 1.4523 (0.0533 @ 0 -> 1.5057 @ 14500); ratio=0.9707; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | PARTIAL | model_loss=1.4615; copy_loss=1.5057; ratio=0.9707; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.067; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.3464; cplus=0.3441; chat=0.353; chat-cplus=0.0088; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 1.3544 @ 0 | 1.5905 @ 14950 | 0.2361 | 0.4754 / 1.2917 / 1.6876 | 1.509 / 1.5921 / 1.6876 (n=100) |
| `L_flow` | 1.1542 @ 0 | 1.5328 @ 14950 | 0.3786 | 0.3961 / 1.2283 / 1.6299 | 1.4505 / 1.5334 / 1.6299 (n=100) |
| `L_var` | 0.4003 @ 0 | 0.016 @ 14950 | -0.3844 | 0.0138 / 0.0222 / 0.4003 | 0.0138 / 0.0163 / 0.0184 (n=100) |
| `L_sigreg` | 0.0417 @ 0 | 0.0024 @ 14950 | -0.0393 | 0.002 / 0.0038 / 0.0417 | 0.002 / 0.0026 / 0.0031 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `L_recon` | 0.9829 @ 0 | 0.3717 @ 14950 | -0.6112 | 0.3656 / 0.3991 / 0.9829 | 0.3656 / 0.3745 / 0.3826 (n=100) |
| `L_recon_pred` | 0.9811 @ 0 | 0.3782 @ 14950 | -0.603 | 0.3728 / 0.4054 / 0.9811 | 0.3728 / 0.383 / 0.3931 (n=100) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 1.0053 @ 14500 | 0.51 | 0.4952 / 0.971 / 1.0074 | 1.0031 / 1.0051 / 1.0074 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.1652 @ 14500 | -0.5586 | 0.0782 / 0.1724 / 0.7239 | 0.1605 / 0.1652 / 0.1694 (n=10) |
| `c_effective_rank` | 9.4728 @ 0 | 50.7635 @ 14500 | 41.2907 | 9.4728 / 37.3294 / 50.7635 | 49.1125 / 50.3122 / 50.7635 (n=10) |
| `c_plus_effective_rank` | 9.2304 @ 0 | 50.4543 @ 14500 | 41.2239 | 9.2304 / 36.6088 / 50.4543 | 48.5324 / 49.8267 / 50.4543 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 5.1663 @ 14500 | -11.5448 | 1.7399 / 6.2113 / 20.4443 | 4.9574 / 5.1049 / 5.1663 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.5596 @ 14500 | -0.4403 | 0.5596 / 0.7103 / 1 | 0.5596 / 0.5639 / 0.5711 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 4.078e-09 @ 14500 | -0.9998 | 4.078e-09 / 0.2068 / 0.9999 | 4.078e-09 / 4.084e-09 / 4.099e-09 (n=10) |
| `coarse_copy_loss` | 0.0533 @ 0 | 1.5057 @ 14500 | 1.4523 | 0.0533 / 1.0681 / 1.5057 | 1.4694 / 1.4945 / 1.5057 (n=10) |
| `coarse_model_loss` | 1.2232 @ 0 | 1.4615 @ 14500 | 0.2383 | 0.4383 / 1.1901 / 1.5515 | 1.3797 / 1.4767 / 1.5515 (n=10) |
| `coarse_batch_mean_loss` | 0.0502 @ 0 | 1.3698 @ 14500 | 1.3196 | 0.0502 / 0.968 / 1.3698 | 1.3222 / 1.3556 / 1.3698 (n=10) |
| `coarse_vs_copy_ratio` | 22.9366 @ 0 | 0.9707 @ 14500 | -21.9659 | 0.9169 / 2.0717 / 22.9366 | 0.9169 / 0.988 / 1.0312 (n=10) |
| `coarse_vs_batch_mean_ratio` | 24.3867 @ 0 | 1.067 @ 14500 | -23.3197 | 1.0098 / 2.234 / 24.3867 | 1.0098 / 1.0893 / 1.1349 (n=10) |
| `L_recon_present` | 0.9858 @ 0 | 0.3464 @ 14500 | -0.6394 | 0.3464 / 0.383 / 0.9858 | 0.3464 / 0.347 / 0.348 (n=10) |
| `L_recon_cplus` | 0.9878 @ 0 | 0.3441 @ 14500 | -0.6436 | 0.3441 / 0.3835 / 0.9878 | 0.3441 / 0.3446 / 0.3455 (n=10) |
| `L_recon_chat` | 0.9878 @ 0 | 0.353 @ 14500 | -0.6349 | 0.3511 / 0.3875 / 0.9878 | 0.3523 / 0.3538 / 0.3556 (n=10) |
| `grad_norm` | 1.5852 @ 0 | 2.2759 @ 14950 | 0.6908 | 0.5325 / 2.424 / 3.1038 | 2.0215 / 2.7504 / 3.08 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=50.7635 @ 14500; `c_cross_video_cosine`=0.1652 @ 14500; `c_std_mean`=1.0053 @ 14500; `coarse_vs_copy_ratio`=0.9707 @ 14500; `coarse_vs_batch_mean_ratio`=1.067 @ 14500; `L_recon_present`=0.3464 @ 14500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 0.9707 and the latest batch-mean ratio is 1.067; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 50.7635, cross-video cosine is 0.1652, and copy loss is 1.5057. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations - fixed-position-decoder

**Status:** FINISHED.
**W&B:** `inv011_fixed_position_decoder` (`io74f32b`), group `inv011_fixed_position_decoder`.
**Commit:** expected `0a9ff46` or later; run used the fixed-position decoder default.
**Command:** see [GUIDE.md](GUIDE.md).
**Full analysis:** [ANALYSIS_inv011_fixed_position_decoder.md](ANALYSIS_inv011_fixed_position_decoder.md).

## Final Metrics

| Metric | Value | Notes |
|---|---:|---|
| `coarse_vs_copy_ratio` | 0.9707 final / 0.9956 plateau | Best recent movement, but gate is <=0.70 |
| `coarse_vs_batch_mean_ratio` | 1.0670 final / 1.0947 plateau | Fails batch-mean gate <=0.50 |
| `coarse_model_loss` | 1.4615 final / 1.4980 plateau | Only slightly below copy at final diag |
| `coarse_copy_loss` | 1.5057 final / 1.5046 plateau | Present and future `c` are separated; not static `c` |
| `c_effective_rank` | 50.76 final / 50.74 plateau | Below target >60 |
| `c_plus_effective_rank` | 50.45 final / 50.40 plateau | EMA target aligned with online rank |
| `c_cross_video_cosine` | 0.1652 final / 0.1655 plateau | Healthy video-specificity |
| `c_std_mean` | 1.0053 final / 1.0051 plateau | Healthy variance |
| `L_recon_present` | 0.3464 final / 0.3465 plateau | Present recon learned under cosine objective |
| `L_recon_cplus` | 0.3441 final / 0.3442 plateau | True-future recon benchmark |
| `L_recon_chat` | 0.3530 final / 0.3541 plateau | Predicted future recon remains close to true future |
| `L_recon_chat - L_recon_cplus` | 0.0088 final / 0.0099 plateau | Gap remains small; recon still blind |
| `grad_skipped` | 0 | Stable; no skip spiral |

## Interpretation Notes

- Compared to `new_recon_loss` (`1u69hpfm`), this run improves copy ratio
  (`1.0652` plateau -> `0.9956`) but loses the rank gate (`60.31` -> `50.74`).
- The run is not a static-`c` trap: `coarse_copy_loss` rises to ~1.5.
- The predictor still fails: copy ratio is near 1 and batch-mean ratio is above 1.
- Reconstruction remains blind: `L_recon_chat - L_recon_cplus` stays around 0.01.
- Verdict: low-rank representation with partial prediction movement; not a passing Phase 1 run.
