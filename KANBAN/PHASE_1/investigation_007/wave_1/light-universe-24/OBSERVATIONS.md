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
   **tied with a low-weight decoder run** ([eager-plant-22](../eager-plant-22/), 0.5845, λ=0.05). If
   weight bound the floor, this run should stand alone at the bottom — it doesn't.
2. `c_effective_rank` is the *lowest* of the five (12.66), and `coarse_vs_copy_ratio` the *highest*
   of the weight runs (1.66) — aggressive reconstruction pressure did **not** make `c` richer or
   prediction better; if anything it traded slightly against both, consistent with the faint `L_flow`
   uptick.

## Connection to the sequence

Top of the weight ladder ([toasty](../toasty-donkey-21/) → [jolly](../jolly-glade-20/) → here). It
closes the weight axis as a dead lever and hands off to the **decoder axis**
([gallant-dew-22](../gallant-dew-22/), [eager-plant-22](../eager-plant-22/)). The λ=1.0 saturation
run ([helpful-snow-25](../../wave_2/helpful-snow-25/)) would have been the next rung but died at
step 200; this run's near-flat slope makes that outcome predictable (~0.581, no break).
