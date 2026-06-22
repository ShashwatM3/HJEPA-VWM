# Run — cerulean-snow-13

## What this run tested

Hypothesis: **`lambda_var=0.5`** (strong variance floor) + **`horizon_k=12`**, no slot loss,
fixes collapse without Goodhart. BRIEF_V0_3 "Run 6".

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Config delta vs centered-slot-k12-run-5

- `lambda_slot`: 0.05 → **0**
- `lambda_var`: 0.10 → **0.5**

## W&B

- Run name: `cerulean-snow-13`
- Project: `smahalanobis-uc-davis/hjepa-vwm`
- URL not pinned in repo; search by display name.

## Parent

[investigation_003](../DESCRIPTION.md) — also informs [investigation_005](../investigation_005/)

## Local log

[`logs/cerulean-snow-13/output.log`](../../../../logs/cerulean-snow-13/output.log) (truncated ~6900 steps in workspace copy)
