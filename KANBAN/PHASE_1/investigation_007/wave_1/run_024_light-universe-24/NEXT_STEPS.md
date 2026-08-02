# Next Steps - run 024 `light-universe-24`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 025 [`earnest-dragon-25`](../../wave_2/run_025_earnest-dragon-25/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — light-universe-24

Closes the weight axis (dead lever). The sequence continues into the **decoder axis**:
[gallant-dew-22](../run_022_gallant-dew-22/) (width) → [eager-plant-22](../run_021_eager-plant-22/) (depth).

The only loose thread off this run is the λ=1.0 saturation point
([helpful-snow-25](../../wave_2/run_029_helpful-snow-25/)), which would confirm the weight axis stays flat at
the extreme — it died at step 200 and is low-priority to re-run given this run's near-zero slope. No
standalone follow-up otherwise.
