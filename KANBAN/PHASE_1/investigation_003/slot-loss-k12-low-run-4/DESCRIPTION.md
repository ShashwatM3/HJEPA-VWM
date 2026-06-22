# Run — slot-loss-k12-low-run-4

**Slug:** BRIEF_V0_3 "Run 4".

## What this run tested

Harder horizon (k=12) with low slot loss — does a harder task + mild slot penalty help?

## Command (approximate)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-slot 0.05
```

## Config delta vs run-3

- `horizon_k`: 4 → **12**
- `lambda_slot`: 0.25 → **0.05**

## W&B

Name not recorded.

## Parent

[investigation_003](../DESCRIPTION.md)
