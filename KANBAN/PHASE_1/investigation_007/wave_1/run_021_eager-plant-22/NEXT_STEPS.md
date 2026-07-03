# Next Steps - run 021 `eager-plant-22`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 022 [`gallant-dew-22`](../run_022_gallant-dew-22/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — eager-plant-22

The terminal run of Wave 1. It retires the decoder axis (paired with
[gallant-dew-22](../run_022_gallant-dew-22/)) and, via its stuck rank + largest blindness gap, motivates the
whole framing of [Wave 2](../../wave_2/DESCRIPTION.md): the floor is `c`-utilization-limited on `d_c`,
so the latent ladder (`n_c`) is the next probe — but expected to fall short.

Directly hands off to [Wave 2 / pious-mountain-28](../../wave_2/run_026_pious-mountain-28/) (n_c=64), the
first latent-axis run. No standalone follow-up.
