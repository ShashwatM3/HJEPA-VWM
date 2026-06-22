# Run — copper-sky-12

**BRIEF Run 5.** Centered slot loss — Goodhart confirmed, slot path rejected.

## What this run tested

After centering fix, does mild slot loss help rank without Goodhart?

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

(`lambda_var=0.10` default.)

## Config delta vs skilled-waterfall-10

- Code: centered `slot_diversity_loss` (commit `ffc33ed`)
- `lambda_cov=0.0027` (active — not logging-only)

## W&B

- Run name: `copper-sky-12`
- Run id: `ejror834`
- Runtime: ~2h 24m (stopped ~step 4300 in chat)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ejror834

## Parent

[investigation_003](../DESCRIPTION.md)
