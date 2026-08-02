# Observations - run 024 `light-universe-24`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `rju7xsh2`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=9250; grad_skipped max=0; grad_has_nan max=0; grad_norm last=3.0062 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0185; dead_dim=0; cross_video_cosine=0.2824 |
| Q3 | Rich latent? | FAIL | c_effective_rank=12.6552; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1818 (0.0533 @ 0 -> 0.2351 @ 9000); ratio=1.6644; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3914; copy_loss=0.2351; ratio=1.6644; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3594; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5855; cplus=0.5843; chat=0.5913; chat-cplus=0.007; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0873 @ 0 | 0.7795 @ 9250 | -2.3078 | 0.7179 / 1.129 / 3.5468 | n/a |
| `L_flow` | 2.8724 @ 0 | 0.4625 @ 9250 | -2.4099 | 0.4002 / 0.8332 / 3.3888 | n/a |
| `L_var` | 0.4299 @ 0 | 0.0048 @ 9250 | -0.4251 | 0.0034 / 0.0181 / 0.4299 | n/a |
| `L_recon` | 1.0264 @ 0 | 0.6293 @ 9250 | -0.3972 | 0.6187 / 0.6657 / 1.0264 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 9250 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 1 @ 9250 | 1 | 0 / 0.8898 / 1 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 1.0185 @ 9000 | 0.5232 | 0.4952 / 0.9255 / 1.0199 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2824 @ 9000 | -0.4414 | 0.1969 / 0.2745 / 0.7239 | n/a |
| `c_effective_rank` | 9.4729 @ 0 | 12.6552 @ 9000 | 3.1823 | 7.9729 / 10.4039 / 12.6552 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 19.3015 @ 9000 | 2.5904 | 16.7111 / 18.5442 / 20.3579 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9435 @ 9000 | -0.0565 | 0.9435 / 0.9748 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.8169 @ 9000 | -0.1829 | 0.8169 / 0.9174 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.2351 @ 9000 | 0.1818 | 0.0533 / 0.2488 / 0.339 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3914 @ 9000 | -2.9192 | 0.3789 / 0.8749 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.0889 @ 9000 | 0.8261 | 0.2627 / 0.9026 / 1.0889 | n/a |
| `coarse_vs_copy_ratio` | 62.0671 @ 0 | 1.6644 @ 9000 | -60.4027 | 1.6112 / 5.9768 / 62.0671 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3594 @ 9000 | -12.2417 | 0.3482 / 1.4535 / 12.6011 | n/a |
| `L_recon_present` | 1.0246 @ 0 | 0.5855 @ 9000 | -0.4391 | 0.5855 / 0.6344 / 1.0246 | n/a |
| `L_recon_cplus` | 1.0251 @ 0 | 0.5843 @ 9000 | -0.4408 | 0.5843 / 0.6371 / 1.0251 | n/a |
| `L_recon_chat` | 1.0266 @ 0 | 0.5913 @ 9000 | -0.4352 | 0.5887 / 0.6431 / 1.0266 | n/a |
| `grad_norm` | 1.4785 @ 0 | 3.0062 @ 9250 | 1.5277 | 0.4971 / 1.9642 / 3.265 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 9250 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.3847 @ 9250 | 0.384 | 6.667e-04 / 0.7233 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=12.6552 @ 9000; `c_cross_video_cosine`=0.2824 @ 9000; `c_std_mean`=1.0185 @ 9000; `coarse_vs_copy_ratio`=1.6644 @ 9000; `coarse_vs_batch_mean_ratio`=0.3594 @ 9000; `L_recon_present`=0.5855 @ 9000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.6644 and the latest batch-mean ratio is 0.3594; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 12.6552, cross-video cosine is 0.2824, and copy loss is 0.2351. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — light-universe-24 (λ_recon = 0.5)

## 2026-06-27 — Final read (~step 9250, plateaued)

| Metric | Value | Read |
|---|---|---|
| `L_recon_present` | **0.5855** | lowest of the weight runs — but only 0.010 below toasty (λ=0.1) at 5× the weight |
| `L_recon_cplus` | 0.5843 | ≈ present |
| `L_recon_chat` | 0.5913 | +0.007 over cplus → recon still blind to prediction |
| `coarse_vs_copy_ratio` | 1.66 | >1: loses to copy (and *higher* than the milder weight runs) |
| `c_effective_rank` | **12.66** | the *lowest* rank of the wave — aggressive recon did not enrich `c` |
| `c_slot_diversity_rank` | 19.3 / 32 | healthy |
| `c_cross_video_cosine` | 0.28 | healthy |
| `L_flow` | 0.42 | faint uptick vs the milder weight runs (~0.40) — the trade-off beginning |

**Trajectory:** identical shape to siblings — warmup drop, asymptote to 0.5855, flat by ~8k.
Stable throughout.

## Interpretation

This run is the weight-bound hypothesis's strongest test, and it **fails it**. Two damning details:
1. Despite 10× the baseline weight, the floor (0.5855) is barely below toasty's (0.596) and is
   **tied with a low-weight decoder run** ([eager-plant-22](../run_021_eager-plant-22/), 0.5845, λ=0.05). If
   weight bound the floor, this run should stand alone at the bottom — it doesn't.
2. `c_effective_rank` is the *lowest* of the five (12.66), and `coarse_vs_copy_ratio` the *highest*
   of the weight runs (1.66) — aggressive reconstruction pressure did **not** make `c` richer or
   prediction better; if anything it traded slightly against both, consistent with the faint `L_flow`
   uptick.

## Connection to the sequence

Top of the weight ladder ([toasty](../run_023_toasty-donkey-21/) → [jolly](../run_020_jolly-glade-20/) → here). It
closes the weight axis as a dead lever and hands off to the **decoder axis**
([gallant-dew-22](../run_022_gallant-dew-22/), [eager-plant-22](../run_021_eager-plant-22/)). The λ=1.0 saturation
run ([helpful-snow-25](../../wave_2/run_029_helpful-snow-25/)) would have been the next rung but died at
step 200; this run's near-flat slope makes that outcome predictable (~0.581, no break).
