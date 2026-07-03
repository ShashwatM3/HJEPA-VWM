# Next Steps - run 002 `efficient-aardvark-2`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use it as smoke evidence only. The correct follow-up is a longer or fuller run that logs enough diagnostic windows to evaluate representation and prediction.

## Linkage To The Research Chain

- This run belongs to `investigation_002`: RunPod, dataloader, W&B, and tiny-data smoke validation.
- Parent investigation next direction: With infrastructure usable, the research shifted to full-data Phase 1 collapse and baseline diagnostics.
- Next chronological W&B run: Run 003 [`comfy-glade-3`](../run_003_comfy-glade-3/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — efficient-aardvark-2

## Why

~1.66 s/step confirmed CPU-bound decode (all frames decoded, then sliced). Fix
required before full SSv2.

## Code change (before next smoke)

Selective decode only needed frame indices (`commit 5e78caa`).

## Spawned

**Next smoke:** [`charmed-haze-4`](../run_004_charmed-haze-4/) — re-time 200 steps post-fix;
metrics must stay bit-identical to this run.

```bash
time python train.py --data ssv2_tiny --steps 200 --log-every 50
```
