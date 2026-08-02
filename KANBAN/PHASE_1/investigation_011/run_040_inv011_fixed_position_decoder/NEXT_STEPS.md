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

## Original Notes Preserved

**Connection:** run 040 reached `coarse_vs_copy_ratio` 0.97 — the best SUSTAINED copy ratio any
full-prediction run in the project has achieved (still short of the <=0.70 gate, and verdict remains
Low-rank rep at rank 50). It showed the fixed-position decoder + cosine loss + residual recipe is
the strongest full-prediction configuration to date but still cannot pass the gate. The read: even
with the honest decoder, prediction is the wall. Attention then turned to a controlled PRESENT-ONLY
study (runs 041-051) to characterize the richest bottleneck B achievable, with the intent of later
transferring that geometry back into a full-prediction run. Full analysis:
[`ANALYSIS_inv011_fixed_position_decoder.md`](ANALYSIS_inv011_fixed_position_decoder.md).
