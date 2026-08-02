# Observations - run 012 `copper-sky-12`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `ejror834`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=killed; step=5650; grad_skipped max=1; grad_has_nan max=0; grad_norm last=7.912e+04 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4585; dead_dim=0; cross_video_cosine=0.7495 |
| Q3 | Rich latent? | FAIL | c_effective_rank=4.8157; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.4862 (0.0533 @ 0 -> 0.5395 @ 5500); ratio=2.0106; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.0847; copy_loss=0.5395; ratio=2.0106; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=4.2299; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9158 @ 0 | 1.176 @ 5650 | -1.7398 | 1.1543 / 1.3272 / 3.4147 | n/a |
| `L_flow` | 2.8663 @ 0 | 1.1186 @ 5650 | -1.7477 | 1.0956 / 1.2763 / 3.3739 | n/a |
| `L_var` | 0.4316 @ 0 | 0.4263 @ 5650 | -0.0053 | 0.0832 / 0.4218 / 0.6355 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 0.4585 @ 5500 | -0.0367 | 0.3645 / 0.4972 / 0.7715 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 5500 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.7495 @ 5500 | 0.0256 | 0.3495 / 0.705 / 0.8424 | n/a |
| `c_effective_rank` | 9.4733 @ 0 | 4.8157 @ 5500 | -4.6576 | 4.5775 / 7.4068 / 12.5126 | n/a |
| `c_slot_diversity_rank` | 16.7119 @ 0 | 20.0728 @ 5500 | 3.3609 | 16.7119 / 22.4059 / 25.96 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 5500 | -4.703e-05 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9994 @ 5500 | -3.975e-04 | 0.9994 / 0.9996 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.5395 @ 5500 | 0.4862 | 0.0533 / 0.544 / 0.8881 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 1.0847 @ 5500 | -2.2259 | 1.0847 / 1.356 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.2564 @ 5500 | -0.0063 | 0.2543 / 0.3747 / 0.6739 | n/a |
| `coarse_vs_copy_ratio` | 62.0662 @ 0 | 2.0106 @ 5500 | -60.0556 | 1.3687 / 7.1501 / 62.0662 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 4.2299 @ 5500 | -8.3712 | 1.9992 / 4.1374 / 12.6011 | n/a |
| `grad_norm` | 0.268 @ 0 | 7.912e+04 @ 5650 | 7.912e+04 | 0.1844 / 3.885e+03 / 1.260e+05 | n/a |
| `grad_skipped` | 0 @ 0 | 1 @ 5650 | 1 | 0 / 0.4386 / 1 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 5500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.7844 @ 5650 | 0.7837 | 6.667e-04 / 0.8093 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=4.8157 @ 5500; `c_cross_video_cosine`=0.7495 @ 5500; `c_std_mean`=0.4585 @ 5500; `coarse_vs_copy_ratio`=2.0106 @ 5500; `coarse_vs_batch_mean_ratio`=4.2299 @ 5500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.0106 and the latest batch-mean ratio is 4.2299; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 4.8157, cross-video cosine is 0.7495, and copy loss is 0.5395. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — copper-sky-12

## Outcome

**Slot lever rejected.** The centered slot loss now works mechanically — it holds slot
diversity high (~20–26/32 the whole run) — yet `c_effective_rank` **collapses** 12.5 →
4.8, `c_cross_video_cosine` climbs to **0.84**, the model never beats copy, and a
4×10⁴ gradient spike with repeated skips appears. Forcing slot diversity actively made
the representation worse: textbook Goodhart. Killed at `_step=5650`.

## Key numbers (verified vs W&B `ejror834`, centered slot, k=12)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `c_std_mean` | `coarse_vs_copy_ratio` | `grad_norm` | `grad_skipped` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.038 | 16.7 | 9.47 | 0.72 | 0.50 | 62.1 | 0.27 | 0 |
| 500 | 0.011 | **26.0** | 12.51 | 0.35 | 0.77 | 3.11 | 2.9 | 0 |
| 1500 | 0.012 | 24.4 | 10.55 | 0.67 | 0.55 | 1.89 | 70.6 | **1** |
| 2000 | 0.011 | 24.3 | 7.35 | **0.78** | 0.43 | 1.37 | 53.9 | **1** |
| 3000 | 0.011 | 23.5 | 6.18 | 0.75 | 0.47 | 2.27 | 139.8 | **1** |
| 4000 | 0.014 | 21.2 | 4.90 | **0.84** | 0.36 | 1.52 | 77.9 | **1** |
| 5000 | 0.015 | 20.0 | 4.67 | 0.80 | 0.41 | 1.71 | **42811** | **1** |
| 5500 | 0.016 | 20.1 | **4.82** | 0.75 | 0.46 | 2.01 | 276 | **1** |

**Correction:** the earlier KANBAN reported rank 13.5/15.1 and slot 7.8/8.0 at steps
3500/4000 with `grad_skipped=0`. Those numbers are not copper-sky-12 — they match
[`serene-cloud-8`](../run_008_serene-cloud-8/)'s trajectory (rank ~13.5/15.1, slot ~7.8/8.0 at
3500/4000). The migration conflated the two slot runs. Corrected here from `ejror834`.

## Interpretation

- **This is the decisive Goodhart evidence.** The centered loss does exactly what it
  says — slot diversity pinned ~20–26 — and it buys nothing: rank still collapses to
  ~4.8 and different videos converge (cosine 0.84). Optimizing the slot metric and
  optimizing the representation are not the same thing. Slot loss is rejected as a
  *training objective*. (The centering fix `ffc33ed` was still worth it — it made the
  rejection trustworthy rather than confounded by a dead loss.)
- **The real binding constraint becomes visible:** `c_std_mean` sits at ~0.4–0.5 the
  whole run — the variance floor at `lambda_var=0.1` is being out-pulled by `L_flow`,
  so the latent shrinks and correlates. This is the observation that motivated the
  pivot to a **stronger variance floor** (the `--lambda-var` flag, `562de1b`).
- **The 10⁴–10⁵-class grad spikes are coupled to the collapse**, not the LR: as slots
  collapse onto a shared direction the bottleneck becomes ill-conditioned and spikes at
  peak LR. Confirmed later by [`cerulean-snow-13`](../run_013_cerulean-snow-13/), which was
  grad-stable at the *same* LR once collapse was fixed — so no LR change was needed.

## Surprises

A slot metric can look great (~24/32) while semantics rot (cosine 0.84) — always pair
slot diversity with `c_cross_video_cosine` and `coarse_vs_copy_ratio`. Source:
W&B `ejror834`; chat ~8700–8780.
