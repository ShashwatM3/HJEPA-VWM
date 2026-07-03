# Next Steps - run 032 `lambda_sigreg_0.3`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_008`: SIGReg ladder on top of reconstruction-capacity recipe.
- Parent investigation next direction: The branch forced a temporal-prediction pivot: test residual prediction and ask whether the model can forecast change rather than memorize static present features.
- Next chronological W&B run: Run 033 [`lambda_sigreg_10.0`](../run_033_lambda_sigreg_10.0/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.
