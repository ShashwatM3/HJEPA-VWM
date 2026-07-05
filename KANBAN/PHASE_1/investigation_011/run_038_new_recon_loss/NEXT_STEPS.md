# Next Steps - run 038 `new_recon_loss`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Healthy rep, no predictor** - Representation health is not the limiting issue; F_c still does not beat copy.

## Immediate Consequence

The representation is good enough that the next question is predictor conditioning/dynamics, not basic c_t collapse. Avoid recommendations that only improve rank.

## Linkage To The Research Chain

- This run belongs to `investigation_011`: cosine reconstruction, fixed-position decoder, present-only geometry sweeps.
- Parent investigation next direction: The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.
- Next chronological W&B run: Run 039 [`original_recon_loss + no-pred`](../run_039_original_recon_loss-no-pred/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** confirmed that better reconstruction geometry (cosine loss) helps the readout but
not the copy gate — reinforcing run 037's lesson that prediction, not representation, is the wall.
The cosine loss became a permanent part of the recipe. Next, investigation_011 attacked the OTHER
half of the reconstruction path: the decoder's ability to store an unconditional content template.
Run 039 isolated present-only reconstruction under the OLD loss with no geometry regularizers
(showing it collapses), and run 040 introduced the fixed-position decoder to remove the template
loophole in a full-prediction run.
