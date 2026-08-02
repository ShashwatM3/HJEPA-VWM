# Next Steps - run 004 `charmed-haze-4`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use it as smoke evidence only. The correct follow-up is a longer or fuller run that logs enough diagnostic windows to evaluate representation and prediction.

## Linkage To The Research Chain

- This run belongs to `investigation_002`: RunPod, dataloader, W&B, and tiny-data smoke validation.
- Parent investigation next direction: With infrastructure usable, the research shifted to full-data Phase 1 collapse and baseline diagnostics.
- Next chronological W&B run: Run 005 [`peachy-terrain-5`](../../investigation_001/run_005_peachy-terrain-5/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — charmed-haze-4

## Why

Post-fix throughput acceptable; metrics unchanged vs pre-fix baseline. Investigation
002 question answered.

## Spawned

**Close** [investigation_002](../DESCRIPTION.md).

**Next training run (different investigation):** [`peachy-terrain-5`](../../investigation_001/run_005_peachy-terrain-5/) in [investigation_001](../../investigation_001/) — first full Phase 1 launch on `ssv2_tiny`, now practical at this s/step.

```bash
python train.py --data ssv2_tiny --steps 30000
```

(Pre-retune hyperparameters; that run crashed — see peachy `NEXT_STEPS`.)
