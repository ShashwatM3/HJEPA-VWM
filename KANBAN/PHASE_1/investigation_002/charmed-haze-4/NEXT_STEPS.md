# Next steps — charmed-haze-4

## Why

Post-fix throughput acceptable; metrics unchanged vs pre-fix baseline. Investigation
002 question answered.

## Spawned

**Close** [investigation_002](../DESCRIPTION.md).

**Next training run (different investigation):** [`peachy-terrain-5`](../../investigation_001/peachy-terrain-5/) in [investigation_001](../../investigation_001/) — first full Phase 1 launch on `ssv2_tiny`, now practical at this s/step.

```bash
python train.py --data ssv2_tiny --steps 30000
```

(Pre-retune hyperparameters; that run crashed — see peachy `NEXT_STEPS`.)
