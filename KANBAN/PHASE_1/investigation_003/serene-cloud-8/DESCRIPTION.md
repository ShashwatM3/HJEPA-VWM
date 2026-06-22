# Run — serene-cloud-8

**BRIEF Run 3.** Aggressive slot-diversity loss at easy horizon.

## What this run tested

Hypothesis: within-video **slot-diversity loss** breaks redundant slots and lifts rank.

## Command

```bash
python train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 \
  --lambda-slot 0.25 --lambda-cov 0.0027
```

(`horizon_k=4` default; `lambda_var=0.10` default.)

## Config delta vs sleek-leaf-7

- `lambda_slot=0.25` (active)
- `lambda_cov=0.0027` (active — not logging-only)

## W&B

- Run name: `serene-cloud-8`
- Run id: `dhp1i3fk`
- Runtime: ~1h 32m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/dhp1i3fk

## Parent

[investigation_003](../DESCRIPTION.md)
