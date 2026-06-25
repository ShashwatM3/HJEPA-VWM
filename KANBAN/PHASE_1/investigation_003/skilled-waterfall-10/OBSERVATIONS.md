# Observations — skilled-waterfall-10

## Headline finding

**`L_slot` glued at ~0.99–1.00 for the entire run** — the gradient did not move the
slot objective at all, even with `lambda_slot=0.05` active. Meanwhile the diagnostic
`c_slot_diversity_rank` slid 16.7 → 3.8, exactly the collapse the loss was supposed to
prevent. The loss was inert.

## Key numbers (verified vs W&B `27i1r9qi`, k=4, raw slot loss)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `coarse_vs_copy_ratio` | `grad_norm` | `grad_skipped` |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 16.7 | 9.47 | 0.72 | 62.1 | 0.27 | 0 |
| 500 | 0.998 | 8.1 | 12.68 | 0.34 | 3.19 | 2.99 | 0 |
| 1000 | 0.998 | 7.3 | 12.39 | 0.51 | 3.60 | 8.86 | 0 |
| 1500 | 1.000 | 6.9 | 10.43 | 0.70 | 1.11 | 55.1 | **1** |
| 2000 | 0.999 | 5.8 | 9.60 | 0.72 | 1.92 | 50.8 | **1** |
| 2500 | 0.993 | 3.8 | 7.29 | 0.58 | 1.90 | 204 | **1** |

Run crashed at `_step=2550`. (`coarse_vs_copy_ratio` near 1 here is **not** progress —
it's the moving-baseline artifact: `copy_loss` rises as the latent drifts, so the ratio
flatters a collapsing model. See the same caution in
[`cerulean-snow-13`](../cerulean-snow-13/) where the ratio became trustworthy only once
the representation stabilized.)

## Root cause (code) — the slot loss/metric mismatch

`slot_diversity_loss` in `losses.py` computed cosine similarity on the **raw** slot
vectors, while the `slot_diversity_rank` diagnostic first **mean-centered** the slots.
A constant DC component shared across all 32 slots makes raw pairwise cosine ≈ 1
regardless of how the slots actually differ in their residual directions — so the loss
sat pinned near 1.0 and its gradient carried almost no information about the metric the
team was trying to move. Fixed by centering the slots in the loss to match the
diagnostic (`commit ffc33ed`, 06-17).

## Interpretation

This is **not** a verdict on slot loss in general — it is a verdict on a buggy
implementation. No `lambda_slot` experiment is interpretable until loss and metric
agree, which is why the centering fix was made **mandatory** before the next slot run
([`copper-sky-12`](../copper-sky-12/), the post-fix rerun). It is also the canonical
"don't just add a CLI flag — read the code" realization the user had pushed for.

Note also the early grad-skips (step 1500+) and the SSH drop ~step 950 in chat: between
the inert loss, the k=4/k=12 launch drift, and the early crash, this run produced no
clean signal on either the horizon or the slot lever.
