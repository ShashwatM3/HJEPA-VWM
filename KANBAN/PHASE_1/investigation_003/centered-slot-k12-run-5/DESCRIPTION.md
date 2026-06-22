# Run — centered-slot-k12-run-5

**Slug:** BRIEF_V0_3 "Run 5".

## What this run tested

After centering fix, does mild slot loss help rank without Goodhart?

## Command (approximate)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-slot 0.05
```

## Config delta vs run-4

- Code: centered `slot_diversity_loss` (commit `ffc33ed`)

## W&B

Name not recorded.

## Parent

[investigation_003](../DESCRIPTION.md)
