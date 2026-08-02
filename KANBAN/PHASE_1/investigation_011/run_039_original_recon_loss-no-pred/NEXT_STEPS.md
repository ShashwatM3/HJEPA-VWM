# Next Steps - run 039 `original_recon_loss + no-pred`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Collapsed rep** - Present reconstruction may improve, but c_t is still weakly spread or video-independent.

## Immediate Consequence

Do not transfer this config directly into prediction. First add or restore geometry pressure, or change bottleneck architecture, until c_t is spread and video-specific.

## Linkage To The Research Chain

- This run belongs to `investigation_011`: cosine reconstruction, fixed-position decoder, present-only geometry sweeps.
- Parent investigation next direction: The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.
- Next chronological W&B run: Run 040 [`inv011_fixed_position_decoder`](../run_040_inv011_fixed_position_decoder/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** the negative control proving present-only reconstruction needs geometry pressure.
It motivates two threads: (a) the fixed-position decoder (run 040/041) to remove the decoder-side
template loophole, and (b) the present-only geometry sweep (runs 042-051) to map which SIGReg x
covariance recipe makes a rich present bottleneck. Much later, run 052 revisits this exact "no
geometry" setup on the sharpened bottleneck (still collapses), and run 053 adds the residual target
to fix the template shortcut specifically.
