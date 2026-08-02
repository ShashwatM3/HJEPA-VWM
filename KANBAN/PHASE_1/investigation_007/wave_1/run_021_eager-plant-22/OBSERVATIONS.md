# Observations - run 021 `eager-plant-22`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `591mt31k`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=9100; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.9479 |
| Q2 | c_t alive / video-specific? | PASS | std=1.037; dead_dim=0; cross_video_cosine=0.2468 |
| Q3 | Rich latent? | FAIL | c_effective_rank=12.9484; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1871 (0.0533 @ 0 -> 0.2405 @ 9000); ratio=1.5278; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3674; copy_loss=0.2405; ratio=1.5278; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3246; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5845; cplus=0.585; chat=0.5959; chat-cplus=0.0109; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0776 @ 0 | 0.4425 @ 9100 | -2.6351 | 0.4135 / 0.8564 / 3.5411 | n/a |
| `L_flow` | 2.8736 @ 0 | 0.4076 @ 9100 | -2.466 | 0.3785 / 0.8195 / 3.3875 | n/a |
| `L_var` | 0.408 @ 0 | 0.0072 @ 9100 | -0.4008 | 0.0036 / 0.0174 / 0.408 | n/a |
| `L_recon` | 1.0379 @ 0 | 0.6271 @ 9100 | -0.4108 | 0.6105 / 0.6542 / 1.0379 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 9100 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 1 @ 9100 | 1 | 0 / 0.888 / 1 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 1.037 @ 9000 | 0.5417 | 0.4952 / 0.9492 / 1.037 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2468 @ 9000 | -0.4771 | 0.1302 / 0.2311 / 0.7239 | n/a |
| `c_effective_rank` | 9.4728 @ 0 | 12.9484 @ 9000 | 3.4756 | 7.837 / 9.8161 / 12.9484 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 17.7663 @ 9000 | 1.0552 | 15.1963 / 16.8345 / 20.3268 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9066 @ 9000 | -0.0934 | 0.9066 / 0.9524 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.7951 @ 9000 | -0.2047 | 0.7951 / 0.8941 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.2405 @ 9000 | 0.1871 | 0.0533 / 0.2628 / 0.3655 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3674 @ 9000 | -2.9432 | 0.3656 / 0.8601 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.1317 @ 9000 | 0.869 | 0.2627 / 0.9601 / 1.1317 | n/a |
| `coarse_vs_copy_ratio` | 62.0672 @ 0 | 1.5278 @ 9000 | -60.5395 | 1.5278 / 5.7754 / 62.0672 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3246 @ 9000 | -12.2765 | 0.3246 / 1.402 / 12.6011 | n/a |
| `L_recon_present` | 1.0398 @ 0 | 0.5845 @ 9000 | -0.4553 | 0.5845 / 0.6251 / 1.0398 | n/a |
| `L_recon_cplus` | 1.0407 @ 0 | 0.585 @ 9000 | -0.4557 | 0.5833 / 0.6316 / 1.0407 | n/a |
| `L_recon_chat` | 1.04 @ 0 | 0.5959 @ 9000 | -0.4441 | 0.588 / 0.6406 / 1.04 | n/a |
| `grad_norm` | 1.4437 @ 0 | 2.9479 @ 9100 | 1.5042 | 0.5379 / 1.9274 / 3.7038 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 9100 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.4017 @ 9100 | 0.4011 | 6.667e-04 / 0.7287 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=12.9484 @ 9000; `c_cross_video_cosine`=0.2468 @ 9000; `c_std_mean`=1.037 @ 9000; `coarse_vs_copy_ratio`=1.5278 @ 9000; `coarse_vs_batch_mean_ratio`=0.3246 @ 9000; `L_recon_present`=0.5845 @ 9000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.5278 and the latest batch-mean ratio is 0.3246; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 12.9484, cross-video cosine is 0.2468, and copy loss is 0.2405. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — eager-plant-22 (decoder 512×4, depth)

## 2026-06-27 — Final read (~step 9100, plateaued)

| Metric | Value | Read |
|---|---|---|
| `L_recon_present` | **0.5845** | **lowest floor of the wave** — yet from a λ=0.05 run, *tying* the 10×-weight run |
| `L_recon_cplus` | 0.5850 | ≈ present |
| `L_recon_chat` | 0.5959 | **+0.0109 over cplus — the largest blindness gap of the wave**, still ≪ the 0.585 floor |
| `coarse_vs_copy_ratio` | 1.53 | lowest of the wave, but still >1 (loses to copy) |
| `c_effective_rank` | 12.95 | ~13 ceiling, unmoved |
| `c_slot_diversity_rank` | 17.8 / 32 | healthy |
| `c_cross_video_cosine` | 0.25 | healthy |
| `L_flow` | 0.42 | baseline-level |

**Trajectory:** warmup drop then asymptote to 0.5845, flat by ~8k (last 1k moves <0.001). Stable
throughout.

## Interpretation

The biggest decoder produces the lowest floor — **but only by 0.005, and from the baseline weight**.
That it *ties the 10× weight run* (light-universe, 0.5855) is the single cleanest proof that neither
axis is the binding constraint: two very different perturbations bottom out at the same ~0.585 wall.

This run is also the **showcase for the blindness finding**. Its `chat − cplus` gap is the wave's
largest (0.0109) — a deeper decoder reconstructs the *true* future latent slightly better, widening
the gap to the *predicted* one — yet 0.011 is still trivial against the 0.585 floor. So even the best
decoder cannot make prediction quality visible to reconstruction. (This is the run plotted in the
W&B report's "present vs cplus vs chat" panel.)

## Connection to the sequence

The last run of [Wave 1](../DESCRIPTION.md) and its decoder ladder's strong end (paired with
[gallant-dew-22](../run_022_gallant-dew-22/)). Its rank stuck at ~13 despite 6× decoder params is the
evidence that crystallized the **utilization-limit** reframe: the under-used axis is `d_c`, not
decoder size — and Wave 2's `n_c` knob doesn't touch `d_c` either. That doubt is what makes
[Wave 2](../../wave_2/DESCRIPTION.md) "mostly confirmatory."
