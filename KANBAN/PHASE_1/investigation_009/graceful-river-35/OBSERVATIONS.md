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
[fine-meadow-34](../fine-meadow-34/)'s static pole. The residual reparametrization works as designed
on the representation (it makes `c` move and stays healthy) but, with recon driving the decorrelation
past the predictive band, `F_c` cannot infer the motion and ties zero. The transient ρ ≈ 0.77 window
at step 3500 — the only point either run beat copy — is the evidence that a governed ρ is the lever to
chase next ([NEXT_STEPS.md](NEXT_STEPS.md)).
