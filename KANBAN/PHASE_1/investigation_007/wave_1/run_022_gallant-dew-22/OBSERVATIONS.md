# Observations - run 022 `gallant-dew-22`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `708jrel8`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=8850; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.8389 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9893; dead_dim=0; cross_video_cosine=0.3045 |
| Q3 | Rich latent? | FAIL | c_effective_rank=12.5094; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1657 (0.0533 @ 0 -> 0.219 @ 8500); ratio=1.7402; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3811; copy_loss=0.219; ratio=1.7402; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3723; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5895; cplus=0.5917; chat=0.5999; chat-cplus=0.0082; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0922 @ 0 | 0.4747 @ 8850 | -2.6176 | 0.4412 / 0.8756 / 3.5382 | n/a |
| `L_flow` | 2.8737 @ 0 | 0.4391 @ 8850 | -2.4346 | 0.4039 / 0.8381 / 3.3846 | n/a |
| `L_var` | 0.4372 @ 0 | 0.0096 @ 8850 | -0.4276 | 0.0026 / 0.0183 / 0.4372 | n/a |
| `L_recon` | 1.0331 @ 0 | 0.616 @ 8850 | -0.4171 | 0.616 / 0.6585 / 1.0331 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 8850 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 1 @ 8850 | 1 | 0 / 0.8848 / 1 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 0.9893 @ 8500 | 0.4941 | 0.4952 / 0.9358 / 1.0175 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 8500 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.3045 @ 8500 | -0.4193 | 0.1266 / 0.2442 / 0.7239 | n/a |
| `c_effective_rank` | 9.4729 @ 0 | 12.5094 @ 8500 | 3.0364 | 7.8527 / 9.5467 / 12.5094 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.2453 @ 8500 | 1.5342 | 15.7724 / 17.0965 / 20.381 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9055 @ 8500 | -0.0944 | 0.9055 / 0.9544 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.8035 @ 8500 | -0.1963 | 0.8035 / 0.9072 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.219 @ 8500 | 0.1657 | 0.0533 / 0.2536 / 0.3243 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3811 @ 8500 | -2.9295 | 0.3811 / 0.8893 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.0238 @ 8500 | 0.7611 | 0.2627 / 0.9197 / 1.0552 | n/a |
| `coarse_vs_copy_ratio` | 62.0673 @ 0 | 1.7402 @ 8500 | -60.3272 | 1.7402 / 6.1101 / 62.0673 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3723 @ 8500 | -12.2288 | 0.3723 / 1.4791 / 12.6011 | n/a |
| `L_recon_present` | 1.0297 @ 0 | 0.5895 @ 8500 | -0.4402 | 0.5885 / 0.6295 / 1.0297 | n/a |
| `L_recon_cplus` | 1.03 @ 0 | 0.5917 @ 8500 | -0.4384 | 0.5891 / 0.6357 / 1.03 | n/a |
| `L_recon_chat` | 1.0308 @ 0 | 0.5999 @ 8500 | -0.4309 | 0.5954 / 0.6443 / 1.0308 | n/a |
| `grad_norm` | 1.4669 @ 0 | 2.8389 @ 8850 | 1.372 | 0.4225 / 1.9035 / 3.3854 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 8850 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 8500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.4304 @ 8850 | 0.4297 | 6.667e-04 / 0.7376 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=12.5094 @ 8500; `c_cross_video_cosine`=0.3045 @ 8500; `c_std_mean`=0.9893 @ 8500; `coarse_vs_copy_ratio`=1.7402 @ 8500; `coarse_vs_batch_mean_ratio`=0.3723 @ 8500; `L_recon_present`=0.5895 @ 8500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.7402 and the latest batch-mean ratio is 0.3723; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 12.5094, cross-video cosine is 0.3045, and copy loss is 0.219. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — gallant-dew-22 (decoder 512×2, width)

## 2026-06-27 — Final read (~step 8500, plateaued)

| Metric | Value | Read |
|---|---|---|
| `L_recon_present` | **0.5895** | ~0.01 below the 0.60 baseline from 3.4× decoder params — negligible |
| `L_recon_cplus` | 0.5917 | ≈ present |
| `L_recon_chat` | 0.5999 | +0.008 over cplus → recon blind to prediction |
| `coarse_vs_copy_ratio` | 1.74 | **highest of the wave** — bigger decoder did not help prediction |
| `c_effective_rank` | 12.51 | ~13 ceiling, unmoved (lowest of the wave) |
| `c_slot_diversity_rank` | 18.2 / 32 | healthy |
| `c_cross_video_cosine` | 0.30 | healthy |
| `L_flow` | 0.44 | baseline-level |

**Trajectory:** warmup drop then asymptote to ~0.5895; logging stopped a little earlier than the
others (~8.5k) but already flat. Stable throughout (`agc_D` clipping low, as expected — `D` never
destabilizes).

## Interpretation

Widening the decoder 3.4× buys ~0.01 of floor — the same trivial magnitude as the entire weight
ladder. This is the first half of the evidence against the **decoder-bound** hypothesis. The tell
that it's a *capacity* miss, not a parameter miss: a wider decoder cannot recover information `c`
never kept (`c_effective_rank` stays ~13/256). Notably `coarse_vs_copy_ratio` is the **highest of
the wave** (1.74) — more decoder did nothing for prediction, consistent with the Wave-level blindness
finding.

## Connection to the sequence

Opens the decoder axis; pairs with [eager-plant-22](../run_021_eager-plant-22/) (depth, the stronger probe).
Width vs depth: depth (eager, 0.5845) edges out width (here, 0.5895) by 0.005 — within noise, and
both far above the 0.55 gate. Together they retire the decoder-bound hypothesis (see
[Wave 1 OBSERVATIONS](../OBSERVATIONS.md)) and leave only the latent axis for [Wave 2](../../wave_2/).
