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

Middle of the weight ladder: [toasty-donkey-21](../toasty-donkey-21/) → **here** →
[light-universe-24](../light-universe-24/). It is the datapoint that turns two endpoints into a
*slope*, and the slope is what formally kills the weight-bound hypothesis. The λ=1.0 saturation run
([helpful-snow-25](../../wave_2/helpful-snow-25/)) was meant to extend this exact ladder — but died
at step 200, so the ladder ends at λ=0.5.
