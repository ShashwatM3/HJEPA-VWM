# Next Steps - run 022 `gallant-dew-22`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 023 [`toasty-donkey-21`](../run_023_toasty-donkey-21/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — gallant-dew-22

Pairs directly with [eager-plant-22](../run_021_eager-plant-22/) (decoder depth) — read them together to
retire the decoder-bound hypothesis. No standalone follow-up: the decoder axis is settled (inert),
and a wider decoder cannot recover information `c` never encoded.

Its low rank (12.51) and high copy-ratio (1.74) feed the Wave-level conclusion that the limit is
`c`'s *utilization* of `d_c`, not decoder size — which is why the next probe is the latent axis in
[Wave 2](../../wave_2/DESCRIPTION.md), and why even that is expected to fall short.
