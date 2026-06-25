# Observations — serene-cloud-8

## Outcome

**First slot-loss run — early Goodhart, on the raw (pre-centering) loss.** Aggressive
`lambda_slot=0.25` did *not* lift slot diversity (the raw loss couldn't, see below);
instead `c_cross_video_cosine` climbed back toward video-independence and the model
stayed far worse than copy. Killed at `_step=4300`. Aggressive slot weight rejected.

## Key numbers (verified vs W&B `dhp1i3fk`, k=4, raw slot loss, lambda_slot=0.25)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `c_std_mean` | `coarse_vs_copy_ratio` | `grad_norm` |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 16.6 | 9.25 | 0.72 | 0.50 | 180.6 | 0.27 |
| 250 | 0.9996 | 10.7 | 12.36 | 0.25 | 0.83 | 2.86 | 0.17 |
| 500 | 0.242 | 6.6 | **15.9** | 0.62 | 0.58 | 1.89 | 2.5 |
| 1000 | 0.195 | 8.3 | 17.9 | 0.66 | 0.55 | 15.2 | 20.5 |
| 2000 | 0.213 | 7.7 | 13.7 | **0.78** | 0.42 | 5.3 | 8.9 |
| 3000 | 0.158 | 7.8 | 13.6 | 0.76 | 0.44 | 9.2 | 24.2 |
| 4250 | 0.188 | 8.0 | 14.5 | 0.74 | 0.45 | 11.5 | 5.0 |

## Interpretation (and corrections)

- **`c_effective_rank` was NOT "worse than run-2."** It actually *rose* to ~16–18 and
  settled ~13–15 — higher than [`sleek-leaf-7`](../sleek-leaf-7/)'s 8.7. The earlier
  KANBAN claim ("rank worse than run-2") is wrong. What's damning is not the rank but
  that **`c_cross_video_cosine` stayed ~0.70–0.78** (not the ~0.84 the old note cited,
  though it does reach ~0.84 later in [`copper-sky-12`](../copper-sky-12/)) and the
  model stayed 5–15× worse than copy. High rank with high cross-video cosine and bad
  copy ratio = the metric moving without the representation improving = **Goodhart**.
- **The slot metric did not actually improve** (`c_slot_diversity_rank` fell 16.6 → ~8).
  On the **raw** loss, `L_slot` does drop from 1.0 to ~0.2 but that is the loss gaming
  its own (uncentered) cosine, not real slot diversity — the same raw/centered mismatch
  later isolated in [`skilled-waterfall-10`](../skilled-waterfall-10/). So serene's
  rejection of `lambda_slot=0.25` was made on a partially-broken objective; the cleaner
  rejection comes after centering (copper-sky-12).
- Training was unstable post-warmup (grad spikes 20–24, `c_std_mean` falling to ~0.45)
  — the same collapse↔instability coupling seen throughout the slot arc.

## What worked

Nothing for production config — but the negative result (aggressive slot weight pushes
the latent toward video-independence) set up the dose-reduction (0.25 → 0.05) and the
horizon discussion that followed. Source: W&B `dhp1i3fk`; chat ~6900–7012.
