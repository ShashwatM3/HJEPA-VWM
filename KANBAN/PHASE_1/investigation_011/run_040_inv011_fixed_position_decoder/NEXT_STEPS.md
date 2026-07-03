# Next Steps - run 040 `inv011_fixed_position_decoder`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_011`: cosine reconstruction, fixed-position decoder, present-only geometry sweeps.
- Parent investigation next direction: The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.
- Next chronological W&B run: Run 041 [`inv011_fixed_position_present_recon`](../run_041_inv011_fixed_position_present_recon/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.
