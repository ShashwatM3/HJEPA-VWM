# Observations — olive-terrain-11

## Outcome

**First clean look at the centered slot loss — and the first sign of Goodhart.** The
centered penalty now *works mechanically* (it holds slot diversity high for a while),
but rank still collapses and cross-video cosine still climbs. Killed at step 3900;
[`copper-sky-12`](../copper-sky-12/) re-ran the identical config and carried the story
to its conclusion.

## Key numbers (verified vs W&B `q40nq0l3`, centered slot, k=12)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `c_std_mean` | `coarse_vs_copy_ratio` | `grad_skipped` |
|---|---|---|---|---|---|---|---|
| 0 | 0.038 | 16.7 | 9.47 | 0.72 | 0.50 | 62.1 | 0 |
| 500 | 0.011 | **25.9** | 12.51 | 0.34 | 0.78 | 3.25 | 0 |
| 1000 | 0.011 | 24.0 | 11.54 | 0.48 | 0.69 | 4.28 | 0 |
| 1500 | 0.011 | 17.0 | 10.74 | 0.67 | 0.55 | 2.85 | **1** |
| 2500 | 0.013 | 10.1 | 9.35 | 0.61 | 0.59 | 2.35 | 0 |
| 3500 | 0.016 | **4.6** | 9.06 | 0.65 | 0.56 | 2.85 | 0 |

## Interpretation

- **The centering fix did its job at the loss level:** `L_slot` now starts ~0.01–0.04
  (vs the inert ~1.0 in [`skilled-waterfall-10`](../skilled-waterfall-10/)) and slot
  diversity *spikes to ~26/32* in the first 500 steps — the penalty genuinely pushes
  slots apart early.
- **But it does not hold, and it does not transfer to the axes we care about.** Slot
  diversity decays 26 → 4.6 over the run despite the active penalty, `c_effective_rank`
  stays ~9–12 (no climb), `c_cross_video_cosine` drifts back up toward 0.65, and
  `c_std_mean` sits at ~0.55 (well under the 1.0 floor — the λ_var=0.1 dose is too weak).
  Forcing slot diversity buys a transient that the rest of the objective erodes.
- This is the **first** Goodhart signal (a working slot loss that still leaves the
  representation collapsed); copper-sky-12 confirms it with a longer run and adds the
  grad-spike failure.

## Mapping note

Resolved from "intermediate / config TBD" via MCP. It is **not** a horizon-only ablation
and **not** a pre-centering run — it is the first centered-slot k=12 slot=0.05 run, the
opening half of the combined "Run 5" slot evidence.
