# Next Steps - run 033 `lambda_sigreg_10.0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Healthy rep, no predictor** - Representation health is not the limiting issue; F_c still does not beat copy.

## Immediate Consequence

The representation is good enough that the next question is predictor conditioning/dynamics, not basic c_t collapse. Avoid recommendations that only improve rank.

## Linkage To The Research Chain

- This run belongs to `investigation_008`: SIGReg ladder on top of reconstruction-capacity recipe.
- Parent investigation next direction: The branch forced a temporal-prediction pivot: test residual prediction and ask whether the model can forecast change rather than memorize static present features.
- Next chronological W&B run: Run 034 [`sigreg-only`](../../investigation_009/run_034_sigreg-only/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** the saturation bookend that made the inv008 conclusion unimpeachable — SIGReg can
drive rank arbitrarily high (73) and prediction only gets worse, so the disease is temporal. This
run's config (SIGReg substrate) is the direct parent of investigation_009 run 034 (`sigreg-only`,
lambda_sigreg=6, full latent, no recon), which reproduced the "rank up, prediction down" law
without any recon confound and drove rho(c_t,c_{t+k}) to ~0.92 (c frozen). The residual-prediction
arm (run 035) then attacked the temporal axis directly.
