# Run — skilled-waterfall-10

**BRIEF Run 4.** Harder horizon + mild slot loss — exposed loss/metric mismatch.

## What this run tested

Harder horizon (k=12) with low slot loss — does a harder task + mild slot penalty help?

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

(`lambda_var=0.10` default.)

## Config delta vs serene-cloud-8

- `horizon_k`: 4 → **12**
- `lambda_slot`: 0.25 → **0.05**
- `lambda_cov=0.0027` (active — not logging-only)

## W&B

- Run name: `skilled-waterfall-10`
- Run id: `27i1r9qi`
- Runtime: ~1h 8m (SSH drop ~step 950 in chat)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/27i1r9qi

## Parent

[investigation_003](../DESCRIPTION.md)
