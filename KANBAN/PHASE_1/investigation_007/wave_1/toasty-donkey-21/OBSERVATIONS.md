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

The low anchor for the weight ladder → [jolly-glade-20](../jolly-glade-20/) (λ=0.2) →
[light-universe-24](../light-universe-24/) (λ=0.5). Read together, these three give the
−0.005/doubling slope that kills the weight-bound hypothesis (see
[Wave 1 OBSERVATIONS](../OBSERVATIONS.md)). Its floor (0.596) is later *beaten by a low-λ decoder
run* ([eager-plant-22](../eager-plant-22/), 0.5845), which is the clinching evidence that the faint
weight trend is noise.
