# Next Steps - run 042 `po_geom_sig7p5_cov0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank decodable** - The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots.

## Immediate Consequence

Treat reconstruction as real but under-constrained. The next run should target rank/slot diversity without sacrificing the reconstruction loss.

## Linkage To The Research Chain

- This run belongs to `investigation_011`: cosine reconstruction, fixed-position decoder, present-only geometry sweeps.
- Parent investigation next direction: The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.
- Next chronological W&B run: Run 043 [`po_geom_sig5_cov0p003`](../run_043_po_geom_sig5_cov0p003/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** one arm of the 10-run present-only geometry sweep (waves 1-2). The sweep's
collective finding — SIGReg + a small covariance penalty drives rank far past 100 (up to ~150 at
sig5/cov0.01, run 047) while keeping videos distinct — proves B+D CAN carry a very rich present
representation, but only with geometry regularizers ON. No full-prediction run has yet inherited
this geometry and passed the copy/batch-mean gates. This directly raised the next question
(investigation_012): can a sharper bottleneck ARCHITECTURE reach this WITHOUT external geometry
regularizers? Run 052 tested that and collapsed, confirming the regularizers are load-bearing.
