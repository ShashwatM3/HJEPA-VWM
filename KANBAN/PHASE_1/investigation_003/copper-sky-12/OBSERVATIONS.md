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
[`serene-cloud-8`](../serene-cloud-8/)'s trajectory (rank ~13.5/15.1, slot ~7.8/8.0 at
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
  peak LR. Confirmed later by [`cerulean-snow-13`](../cerulean-snow-13/), which was
  grad-stable at the *same* LR once collapse was fixed — so no LR change was needed.

## Surprises

A slot metric can look great (~24/32) while semantics rot (cosine 0.84) — always pair
slot diversity with `c_cross_video_cosine` and `coarse_vs_copy_ratio`. Source:
W&B `ejror834`; chat ~8700–8780.
