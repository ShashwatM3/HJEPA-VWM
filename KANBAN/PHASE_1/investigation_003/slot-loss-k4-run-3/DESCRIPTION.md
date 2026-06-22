# Run — slot-loss-k4-run-3

**Slug:** BRIEF_V0_3 "Run 3" — no W&B name in repo.

## What this run tested

Hypothesis: within-video **slot-diversity loss** breaks redundant slots and lifts rank.

## Command (approximate)

```bash
python train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 \
  --lambda-slot 0.25 --lambda-cov 0.0027
```

(`horizon_k=4` default at launch.)

## Config delta vs run-2

- `lambda_slot=0.25`
- `horizon_k=4` (easy horizon)

## W&B

Name not recorded.

## Parent

[investigation_003](../DESCRIPTION.md)
