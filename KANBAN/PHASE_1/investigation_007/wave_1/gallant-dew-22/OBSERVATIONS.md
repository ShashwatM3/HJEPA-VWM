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

Opens the decoder axis; pairs with [eager-plant-22](../eager-plant-22/) (depth, the stronger probe).
Width vs depth: depth (eager, 0.5845) edges out width (here, 0.5895) by 0.005 — within noise, and
both far above the 0.55 gate. Together they retire the decoder-bound hypothesis (see
[Wave 1 OBSERVATIONS](../OBSERVATIONS.md)) and leave only the latent axis for [Wave 2](../../wave_2/).
