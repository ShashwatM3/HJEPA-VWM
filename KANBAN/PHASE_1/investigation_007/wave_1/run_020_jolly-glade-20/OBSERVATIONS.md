# Observations - run 020 `jolly-glade-20`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `5x7aoxnn`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=9050; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.908 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0427; dead_dim=0; cross_video_cosine=0.2506 |
| Q3 | Rich latent? | FAIL | c_effective_rank=12.9849; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1946 (0.0533 @ 0 -> 0.2479 @ 9000); ratio=1.5638; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3877; copy_loss=0.2479; ratio=1.5638; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3383; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5915; cplus=0.5906; chat=0.5966; chat-cplus=0.0061; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0873 @ 0 | 0.5632 @ 9050 | -2.5241 | 0.5202 / 0.9581 / 3.5392 | n/a |
| `L_flow` | 2.8724 @ 0 | 0.433 @ 9050 | -2.4394 | 0.3905 / 0.8336 / 3.3888 | n/a |
| `L_var` | 0.4299 @ 0 | 0.007 @ 9050 | -0.4229 | 0.0035 / 0.0186 / 0.4299 | n/a |
| `L_recon` | 1.0264 @ 0 | 0.6338 @ 9050 | -0.3926 | 0.6242 / 0.6705 / 1.0264 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 9050 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 1 @ 9050 | 1 | 0 / 0.8874 / 1 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 1.0427 @ 9000 | 0.5474 | 0.4952 / 0.9277 / 1.0427 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2506 @ 9000 | -0.4733 | 0.1863 / 0.2731 / 0.7239 | n/a |
| `c_effective_rank` | 9.4729 @ 0 | 12.9849 @ 9000 | 3.5119 | 8.0082 / 10.2721 / 12.9849 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.7778 @ 9000 | 2.0668 | 16.3093 / 18.0175 / 20.3421 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9231 @ 9000 | -0.0768 | 0.9231 / 0.963 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.8163 @ 9000 | -0.1835 | 0.8163 / 0.9109 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.2479 @ 9000 | 0.1946 | 0.0533 / 0.2536 / 0.3436 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3877 @ 9000 | -2.9229 | 0.3877 / 0.8684 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.1459 @ 9000 | 0.8832 | 0.2627 / 0.9156 / 1.1459 | n/a |
| `coarse_vs_copy_ratio` | 62.0671 @ 0 | 1.5638 @ 9000 | -60.5033 | 1.5638 / 5.9168 / 62.0671 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3383 @ 9000 | -12.2628 | 0.3383 / 1.4376 / 12.6011 | n/a |
| `L_recon_present` | 1.0246 @ 0 | 0.5915 @ 9000 | -0.4331 | 0.591 / 0.6381 / 1.0246 | n/a |
| `L_recon_cplus` | 1.0251 @ 0 | 0.5906 @ 9000 | -0.4345 | 0.5906 / 0.6416 / 1.0251 | n/a |
| `L_recon_chat` | 1.0266 @ 0 | 0.5966 @ 9000 | -0.4299 | 0.5949 / 0.6477 / 1.0266 | n/a |
| `grad_norm` | 1.4785 @ 0 | 2.908 @ 9050 | 1.4295 | 0.4981 / 1.9629 / 4.027 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 9050 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.4075 @ 9050 | 0.4068 | 6.667e-04 / 0.7305 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=12.9849 @ 9000; `c_cross_video_cosine`=0.2506 @ 9000; `c_std_mean`=1.0427 @ 9000; `coarse_vs_copy_ratio`=1.5638 @ 9000; `coarse_vs_batch_mean_ratio`=0.3383 @ 9000; `L_recon_present`=0.5915 @ 9000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.5638 and the latest batch-mean ratio is 0.3383; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 12.9849, cross-video cosine is 0.2506, and copy loss is 0.2479. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — jolly-glade-20 (λ_recon = 0.2)

## 2026-06-27 — Final read (~step 9050, plateaued)

| Metric | Value | Read |
|---|---|---|
| `L_recon_present` | **0.5915** | between toasty (0.596) and light (0.586), as a clean slope would predict |
| `L_recon_cplus` | 0.5906 | ≈ present |
| `L_recon_chat` | 0.5966 | +0.006 over cplus → recon blind to prediction |
| `coarse_vs_copy_ratio` | 1.56 | >1: loses to copy |
| `c_effective_rank` | 12.98 | ~13 ceiling, unmoved |
| `c_slot_diversity_rank` | 18.8 / 32 | healthy |
| `c_cross_video_cosine` | 0.25 | healthy |
| `L_flow` | 0.40 | baseline-level (no degradation at 4× weight yet) |

**Trajectory:** same shape as its siblings — fast drop through warmup, asymptote to ~0.5915.
Stable throughout.

## Interpretation

The mid-ladder point lands exactly where a monotone weight effect predicts (0.5915, between 0.596
and 0.586). It confirms the weight axis is **real but trivially weak**: the full 0.1→0.2→0.5 ladder
moves the floor only ~0.010 total — roughly **−0.005 per weight-doubling**. Extrapolated, reaching
the 0.55 gate would need λ≈30. `L_flow` (0.40) shows no prediction degradation at 4× weight, so the
weight cost the SWEEP_PLAN warned about hasn't bitten yet (it begins, faintly, at λ=0.5 / the
λ=1.0 saturation run).

## Connection to the sequence

Middle of the weight ladder: [toasty-donkey-21](../run_023_toasty-donkey-21/) → **here** →
[light-universe-24](../run_024_light-universe-24/). It is the datapoint that turns two endpoints into a
*slope*, and the slope is what formally kills the weight-bound hypothesis. The λ=1.0 saturation run
([helpful-snow-25](../../wave_2/run_029_helpful-snow-25/)) was meant to extend this exact ladder — but died
at step 200, so the ladder ends at λ=0.5.
