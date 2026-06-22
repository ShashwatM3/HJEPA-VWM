# Run — cerulean-snow-13

## What this run tested

Hypothesis: **`lambda_var=0.5`** (strong variance floor) + **`horizon_k=12`**, no slot loss,
fixes collapse without Goodhart. BRIEF_V0_3 "Run 6".

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Config delta vs copper-sky-12

- `lambda_slot`: 0.05 → **0**
- `lambda_var`: 0.10 → **0.5**

## W&B

- Run name: `cerulean-snow-13`
- Run id: `4lo4j7qb`
- Runtime: ~3h 7m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4lo4j7qb

## Parent

[investigation_003](../DESCRIPTION.md) — also informs [investigation_005](../investigation_005/)

## Local log

[`logs/cerulean-snow-13/output.log`](../../../../logs/cerulean-snow-13/output.log) (truncated ~6900 steps in workspace copy)
