# Observations - run 035 `sigreg-recon-residual`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `jsh6uo7p`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=14050; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.2538 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0048; dead_dim=0; cross_video_cosine=0.1699 |
| Q3 | Rich latent? | PARTIAL | c_effective_rank=57.9796; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | PARTIAL | copy_loss trend=up 1.5063 (0.0533 @ 0 -> 1.5596 @ 14000); ratio=1.0883; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.6973; copy_loss=1.5596; ratio=1.0883; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.1757; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5691; cplus=0.5671; chat=0.5821; chat-cplus=0.015; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 1.6052 @ 0 | 1.7921 @ 14050 | 0.187 | 0.3887 / 1.3173 / 1.8608 | 1.6117 / 1.7382 / 1.8608 (n=82) |
| `L_flow` | 1.1793 @ 0 | 1.7092 @ 14050 | 0.53 | 0.2822 / 1.2284 / 1.7778 | 1.53 / 1.6567 / 1.7778 (n=82) |
| `L_var` | 0.4081 @ 0 | 0.0185 @ 14050 | -0.3896 | 0.0147 / 0.0234 / 0.4081 | 0.0147 / 0.0166 / 0.0185 (n=82) |
| `L_sigreg` | 0.0444 @ 0 | 0.0025 @ 14050 | -0.0419 | 0.0018 / 0.0038 / 0.0444 | 0.0018 / 0.0023 / 0.003 (n=82) |
| `L_recon` | 1.0379 @ 0 | 0.6082 @ 14050 | -0.4297 | 0.6003 / 0.6344 / 1.0379 | 0.601 / 0.6093 / 0.6184 (n=82) |
| `L_recon_pred` | 1.0372 @ 0 | 0.6196 @ 14050 | -0.4177 | 0.6098 / 0.6422 / 1.0372 | 0.6098 / 0.6202 / 0.631 (n=82) |
| `recon_scale` | 0 @ 0 | 1 @ 14050 | 1 | 0 / 0.9273 / 1 | 1 / 1 / 1 (n=82) |
| `c_std_mean` | 0.4952 @ 0 | 1.0048 @ 14000 | 0.5096 | 0.4952 / 0.9668 / 1.0056 | 1.0036 / 1.0044 / 1.0056 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.1699 @ 14000 | -0.554 | 0.0843 / 0.1739 / 0.7239 | 0.1681 / 0.1697 / 0.1721 (n=9) |
| `c_effective_rank` | 9.4729 @ 0 | 57.9796 @ 14000 | 48.5067 | 9.4729 / 39.8638 / 57.9796 | 55.8959 / 57.2592 / 57.9796 (n=9) |
| `c_slot_diversity_rank` | 16.711 @ 0 | 6.2837 @ 14000 | -10.4273 | 3.5819 / 7.8418 / 19.2282 | 6.0518 / 6.209 / 6.2837 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6063 @ 14000 | -0.3937 | 0.6063 / 0.7597 / 0.9999 | 0.6063 / 0.6123 / 0.6222 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 7.732e-09 @ 14000 | -0.9998 | 4.134e-09 / 0.2827 / 0.9999 | 7.402e-09 / 1.219e-08 / 2.942e-08 (n=9) |
| `coarse_copy_loss` | 0.0533 @ 0 | 1.5596 @ 14000 | 1.5063 | 0.0533 / 1.017 / 1.5596 | 1.4667 / 1.5243 / 1.5596 (n=9) |
| `coarse_model_loss` | 1.225 @ 0 | 1.6973 @ 14000 | 0.4723 | 0.3714 / 1.2117 / 1.7318 | 1.561 / 1.6352 / 1.7318 (n=9) |
| `coarse_batch_mean_loss` | 0.0502 @ 0 | 1.4436 @ 14000 | 1.3935 | 0.0502 / 0.9363 / 1.4436 | 1.3555 / 1.4104 / 1.4436 (n=9) |
| `coarse_vs_copy_ratio` | 22.9689 @ 0 | 1.0883 @ 14000 | -21.8807 | 0.9149 / 2.1127 / 22.9689 | 1.0143 / 1.0728 / 1.1195 (n=9) |
| `coarse_vs_batch_mean_ratio` | 24.4211 @ 0 | 1.1757 @ 14000 | -23.2454 | 0.9863 / 2.2749 / 24.4211 | 1.0948 / 1.1594 / 1.209 (n=9) |
| `L_recon_present` | 1.0398 @ 0 | 0.5691 @ 14000 | -0.4707 | 0.5691 / 0.6022 / 1.0398 | 0.5691 / 0.5699 / 0.571 (n=9) |
| `L_recon_cplus` | 1.0407 @ 0 | 0.5671 @ 14000 | -0.4736 | 0.5671 / 0.6013 / 1.0407 | 0.5671 / 0.5678 / 0.569 (n=9) |
| `L_recon_chat` | 1.0405 @ 0 | 0.5821 @ 14000 | -0.4583 | 0.5757 / 0.6086 / 1.0405 | 0.5783 / 0.5801 / 0.5838 (n=9) |
| `grad_norm` | 3.0617 @ 0 | 2.2538 @ 14050 | -0.8079 | 0.6638 / 2.2073 / 3.0617 | 2.219 / 2.6431 / 2.8326 (n=82) |
| `grad_skipped` | 0 @ 0 | 0 @ 14050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=82) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0122 @ 14050 | 0.0115 | 6.667e-04 / 0.5317 / 1 | 0.0122 / 0.1295 / 0.302 (n=82) |

## Last Key Metrics

`c_effective_rank`=57.9796 @ 14000; `c_cross_video_cosine`=0.1699 @ 14000; `c_std_mean`=1.0048 @ 14000; `coarse_vs_copy_ratio`=1.0883 @ 14000; `coarse_vs_batch_mean_ratio`=1.1757 @ 14000; `L_recon_present`=0.5691 @ 14000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.0883 and the latest batch-mean ratio is 1.1757; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 57.9796, cross-video cosine is 0.1699, and copy loss is 1.5596. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch cleaned optimizer and regularization settings around residual prediction to see whether the result survived a full clean run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — graceful-river-35 (Run 2: residual prediction + recon)

## 2026-06-28 — Final read (manually ended, step 14050; plateau = mean of last 3 diag points)

All values pulled from W&B (`jsh6uo7p`, full history, 29 diag points). Cross-run synthesis and the
derived ρ(c_t, c_{t+k}) lens live in [../RESULTS_ANALYSIS.md](../RESULTS_ANALYSIS.md). Absolute
`L_flow` / `coarse_copy_loss` are at the natural Δ scale here, so read the **ratio** (not the
absolutes) across the two parametrizations.

| Metric | Value | Trajectory |
|---|---|---|
| `coarse_vs_copy_ratio` | **1.08** (osc ~1.05) | 23 → 1.17 (2500) → **0.915 (3500)** → settles 1.01–1.12 |
| `c_effective_rank` | **57.9** | climbs the whole run, still rising at the cut |
| `coarse_copy_loss` = ‖Δ‖² | **1.554** | **rises** 0.20 → 1.554 (opposite sign to Run 1) |
| `coarse_model_loss` | 1.682 | tracks copy upward, ends *slightly above* it |
| implied ρ(c_t, c_{t+k}) | **~0.23** | falls ~0.88 → ~0.23 (derived, see RESULTS_ANALYSIS §5) |
| `coarse_vs_batch_mean_ratio` | 1.17 | |
| `L_flow` | 1.65 | Δ-scale — not comparable to Run 1 |
| `c_std_mean` | 1.005 | reaches 1.0, var floor satisfied (`L_var` ~0.016, barely active) |
| `c_cross_video_cosine` | **0.170** | low and stable (min 0.084 @2000) |
| `c_slot_diversity_rank` | 6.28 / 32 | dips to ~3.6, recovers to 6.3 |
| `c_attn_entropy` / `_min` | 0.607 / ~1e-8 | sharpens; ≥1 slot attends ~a single token |
| `L_recon_present` | **0.569** | falls just below the inv007 0.585 floor, still gently dropping |
| `L_recon_chat` − `L_recon_cplus` | 0.582 − 0.567 = **0.015** | gap stays trivial |
| `L_var` / `L_sigreg` | 0.016 / 0.0025 | |
| `grad_skipped` / `instability_warn` / `grad_has_nan` / `c_dead_dim_frac` | 0 / 0 / 0 / 0 | every step |

## Interpretation (metric-level)

- **The ratio fell ~6× vs the substrate arm, but read the components, not the headline.** `coarse_vs_copy_ratio`
  drops from 23 to a minimum of **0.915 at step 3500** (where `coarse_model_loss` 0.371 < `coarse_copy_loss`
  0.406 — `F_c` genuinely beat copy), then oscillates 1.01–1.12 for the entire back half. Because
  `coarse_model_loss` (1.682) ends *slightly above* `coarse_copy_loss` (1.554), the terminal ratio > 1
  means `F_c`'s predicted residual `Δ̂` is on average marginally *worse* than predicting `Δ̂ = 0`.
  Read together, the metrics describe `F_c` reverting to "predict no change" — a **tie at zero**, not
  skillful prediction, except for the one transient window at step 3500.
- **`coarse_copy_loss` moved in the opposite direction to Run 1.** ‖Δ‖² *rises* 0.20 → 1.554 (implied
  ρ → ~0.23): the future latent decorrelated from the present, so `c` now moves a lot over the
  horizon. This is the inverse of the static-`c` signature and the first run in the project where the
  copy baseline became *non-trivial* by `c` moving rather than freezing.
- **The representation metrics are the healthiest in the wave.** `c_effective_rank` reaches **57.9**
  and is still climbing — higher than Run 1's 50 at *lower* λ_sigreg — while `c_cross_video_cosine`
  stays low at 0.17 (no cross-video drift) and `c_std_mean` holds 1.0 with the var floor only barely
  active (the recon supplies the variance work, so the floor is not load-bearing here, unlike Run 1).
  Bottleneck attention sharpens hard (`c_attn_entropy` → 0.61, `_min` → ~1e-8). Slot redundancy
  persists (`c_slot_diversity_rank` 6.3/32), as in Run 1.
- **Reconstruction floor nudged, blindness intact.** `L_recon_present` (0.569) drops just below the
  inv007 0.585 floor — the first lever to move it — but only by ~0.015 (≈41.5% → 43% of variance).
  `L_recon_chat` (0.582) ≈ `L_recon_cplus` (0.567), gap **0.015**: decoding the *predicted* future
  `ĉ = c_t + Δ̂` reconstructs `e_{t+k}` essentially as well as decoding the *true* future, so recon
  remains insensitive to prediction quality.

## Interpretation w.r.t. the registered hypothesis

- **Q1 — does the residual drive the ratio toward < 1?** Partially. The ratio crossed below 1 at step
  3500 (0.915) and fell ~6× overall, but did not settle below 1; it ties copy from above (~1.05–1.08).
  The pre-registered **tie-by-zero risk (R2-P2, `Δ̂ → 0`) materialized.**
- **Q2 — does reconstruction-with-residuals help?** Reads as **not supported** on R2-P3: `L_recon_chat`
  ≈ `L_recon_cplus` (gap 0.015), the blindness finding holds; the residual decode did not make
  prediction quality visible to reconstruction.
- **R2-P1 (rank ~45–50):** exceeded — **57.9**, the richest `c` in the wave.
- **R2-P2 (ratio ≥ 1, watch for a dip below baseline):** reads as a **match on both clauses** — back-half
  oscillation ~1.05 *and* the flagged transient dip to 0.915.
- **R2-P4 (stability fine; watch `Δ̂ → 0` and a climbing cosine):** stability perfect (0 skips). The
  `Δ̂ → 0` collapse-watch **occurred**; the cosine collapse-watch did **not** — `c_cross_video_cosine`
  stayed *low* (0.17), the opposite of the flagged direction.
- **Not pre-registered:** the residual objective reversed the static-`c` signature at the
  representation level (ρ collapse 0.88 → 0.23, rank 58, healthy cosine/std), relocating the
  bottleneck from "`c` is static" to "`c`'s motion is not predictable from `c_t`".

## Connection to the wave

This is the **dynamic-ρ pole** of [investigation_009](../DESCRIPTION.md), the mirror image of
[fine-meadow-34](../run_034_sigreg-only/)'s static pole. The residual reparametrization works as designed
on the representation (it makes `c` move and stays healthy) but, with recon driving the decorrelation
past the predictive band, `F_c` cannot infer the motion and ties zero. The transient ρ ≈ 0.77 window
at step 3500 — the only point either run beat copy — is the evidence that a governed ρ is the lever to
chase next ([NEXT_STEPS.md](NEXT_STEPS.md)).
