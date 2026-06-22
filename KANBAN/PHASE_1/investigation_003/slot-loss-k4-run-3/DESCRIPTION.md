# Run — slot-loss-k4-run-3

**Slug:** BRIEF_V0_3 "Run 3" — no W&B name in repo.

## What this run tested

Hypothesis: within-video **slot-diversity loss** breaks redundant slots and lifts rank.

## Command (approximate)

```bash
python train.py --data ssv2 --steps 15000 --lambda-slot 0.25 --horizon-k 4
```

(`lambda_cov` may have been nonzero in some launch docs — primary delta is slot + k=4.)

## Config delta vs run-2

- `lambda_slot=0.25`
- `horizon_k=4` (easy horizon)

## W&B

Name not recorded.

## Parent

[investigation_003](../DESCRIPTION.md)
