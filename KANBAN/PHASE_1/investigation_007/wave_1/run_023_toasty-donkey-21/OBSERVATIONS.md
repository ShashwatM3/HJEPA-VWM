# Observations - run 023 `toasty-donkey-21`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `a2trqp9c`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=9200; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.9442 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0345; dead_dim=0; cross_video_cosine=0.2487 |
| Q3 | Rich latent? | FAIL | c_effective_rank=12.9731; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1842 (0.0533 @ 0 -> 0.2375 @ 9000); ratio=1.5857; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3766; copy_loss=0.2375; ratio=1.5857; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3422; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5959; cplus=0.5954; chat=0.6027; chat-cplus=0.0073; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0873 @ 0 | 0.4686 @ 9200 | -2.6188 | 0.4496 / 0.8931 / 3.5367 | n/a |
| `L_flow` | 2.8724 @ 0 | 0.4008 @ 9200 | -2.4715 | 0.3815 / 0.8263 / 3.3888 | n/a |
| `L_var` | 0.4299 @ 0 | 0.0052 @ 9200 | -0.4247 | 0.0035 / 0.0176 / 0.4299 | n/a |
| `L_recon` | 1.0264 @ 0 | 0.6512 @ 9200 | -0.3752 | 0.6277 / 0.6728 / 1.0264 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 9200 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 1 @ 9200 | 1 | 0 / 0.8892 / 1 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 1.0345 @ 9000 | 0.5392 | 0.4952 / 0.9273 / 1.0345 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2487 @ 9000 | -0.4752 | 0.1862 / 0.267 / 0.7239 | n/a |
| `c_effective_rank` | 9.4729 @ 0 | 12.9731 @ 9000 | 3.5002 | 8.0355 / 10.0999 / 12.9731 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.8794 @ 9000 | 2.1684 | 15.7372 / 17.6274 / 20.3239 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9084 @ 9000 | -0.0916 | 0.9084 / 0.9556 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.8013 @ 9000 | -0.1985 | 0.8013 / 0.9038 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.2375 @ 9000 | 0.1842 | 0.0533 / 0.2534 / 0.3565 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3766 @ 9000 | -2.934 | 0.3757 / 0.869 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.1006 @ 9000 | 0.8379 | 0.2627 / 0.9168 / 1.1006 | n/a |
| `coarse_vs_copy_ratio` | 62.0671 @ 0 | 1.5857 @ 9000 | -60.4814 | 1.5857 / 5.9285 / 62.0671 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3422 @ 9000 | -12.2589 | 0.3422 / 1.4358 / 12.6011 | n/a |
| `L_recon_present` | 1.0246 @ 0 | 0.5959 @ 9000 | -0.4287 | 0.5954 / 0.6413 / 1.0246 | n/a |
| `L_recon_cplus` | 1.0251 @ 0 | 0.5954 @ 9000 | -0.4297 | 0.5954 / 0.6449 / 1.0251 | n/a |
| `L_recon_chat` | 1.0266 @ 0 | 0.6027 @ 9000 | -0.4239 | 0.5991 / 0.6507 / 1.0266 | n/a |
| `grad_norm` | 1.4785 @ 0 | 2.9442 @ 9200 | 1.4658 | 0.4987 / 1.9448 / 3.4408 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 9200 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 9000 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.3904 @ 9200 | 0.3897 | 6.667e-04 / 0.7251 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=12.9731 @ 9000; `c_cross_video_cosine`=0.2487 @ 9000; `c_std_mean`=1.0345 @ 9000; `coarse_vs_copy_ratio`=1.5857 @ 9000; `coarse_vs_batch_mean_ratio`=0.3422 @ 9000; `L_recon_present`=0.5959 @ 9000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.5857 and the latest batch-mean ratio is 0.3422; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 12.9731, cross-video cosine is 0.2487, and copy loss is 0.2375. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — toasty-donkey-21 (λ_recon = 0.1)

## 2026-06-27 — Final read (~step 9200, plateaued)

| Metric | Value | Read |
|---|---|---|
| `L_recon_present` | **0.5959** | floor barely below baseline ~0.60; nowhere near the 0.55 gate |
| `L_recon_cplus` | 0.5954 | ≈ present → future-from-true-latent as hard as present |
| `L_recon_chat` | 0.6027 | only +0.007 over cplus → prediction nearly invisible to recon |
| `coarse_vs_copy_ratio` | 1.59 | >1: loses to copy baseline |
| `c_effective_rank` | 12.97 | the ~13/256 ceiling, unmoved |
| `c_slot_diversity_rank` | 18.9 / 32 | healthy |
| `c_cross_video_cosine` | 0.25 | healthy (no Mode-A collapse) |
| `L_flow` | 0.41 | baseline-level |

**Trajectory:** `L_recon_present` dropped 1.02 → ~0.61 during the 2000-step recon warmup, then crept
asymptotically to 0.596 and flattened (last 2k steps move <0.002). Stable throughout
(`grad_skipped`=0, `instability_warn`=0).

## Interpretation

This is the **highest floor of the wave** — i.e. the *smallest* λ moved the floor the *least*, as a
weight-bound hypothesis would predict. But the effect is tiny (0.596 vs the 0.585 reached at λ=0.5),
and the blindness gap (chat−cplus = 0.007) is already negligible here. So even at the low anchor, the
signal that "weight matters" is barely above noise.

## Connection to the sequence

The low anchor for the weight ladder → [jolly-glade-20](../run_020_jolly-glade-20/) (λ=0.2) →
[light-universe-24](../run_024_light-universe-24/) (λ=0.5). Read together, these three give the
−0.005/doubling slope that kills the weight-bound hypothesis (see
[Wave 1 OBSERVATIONS](../OBSERVATIONS.md)). Its floor (0.596) is later *beaten by a low-λ decoder
run* ([eager-plant-22](../run_021_eager-plant-22/), 0.5845), which is the clinching evidence that the faint
weight trend is noise.
