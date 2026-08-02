# Next Steps - run 030 `lambda_sigreg_3.0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_008`: SIGReg ladder on top of reconstruction-capacity recipe.
- Parent investigation next direction: The branch forced a temporal-prediction pivot: test residual prediction and ask whether the model can forecast change rather than memorize static present features.
- Next chronological W&B run: Run 031 [`lambda_sigreg_1.0`](../run_031_lambda_sigreg_1.0/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** one arm of the 4-wide lambda_sigreg sweep {0.3, 1.0, 3.0, 10.0}. Read together
with the other arms, not alone. The wave conclusion — rank rises monotonically with lambda
(13 -> 34 -> 73) while copy ratio worsens monotonically (2.2 -> 4.7 -> 8.2) — is decisive
evidence the disease is TEMPORAL, not utilization, and forced the residual-prediction pivot in
investigation_009 (run 034 SIGReg substrate, run 035 residual). Do not build prediction on this
config: its c is rich but static.
