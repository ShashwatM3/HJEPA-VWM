# Next Steps - run 020 `jolly-glade-20`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 021 [`eager-plant-22`](../run_021_eager-plant-22/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — jolly-glade-20

Continues at [light-universe-24](../run_024_light-universe-24/) (λ=0.5, the aggressive end of the weight
ladder). Together with [toasty-donkey-21](../run_023_toasty-donkey-21/) these three settle the weight axis
(inert) — see [Wave 1 OBSERVATIONS](../OBSERVATIONS.md).

The natural extension of this ladder, λ=1.0 ([helpful-snow-25](../../wave_2/run_029_helpful-snow-25/)), was
scheduled in Wave 2 but never produced data (died at step 200). If the weight axis is ever revisited,
that is the one missing rung — though Wave 1's slope already makes its outcome (~0.581, no break)
near-certain. No standalone follow-up for this config.
