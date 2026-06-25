# Run — copper-sky-12

**BRIEF Run 5.** Centered slot loss — Goodhart confirmed, slot path rejected.

## What this run tested

After the centering fix, does mild slot loss help rank without Goodhart — run **longer**
than [`olive-terrain-11`](../olive-terrain-11/) (which showed the same config Goodharting
but was killed at 3900)? This is the **re-run** of olive-terrain-11 with an identical
config; together the two are the combined "Run 5" slot evidence.

## Command (verified vs W&B config `ejror834`)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

Config confirmed: `horizon_k=12`, `lambda_slot=0.05`, `lambda_cov=0.0027`,
`lambda_var=0.10` (default). Launched 06-19 09:40 UTC (chat ~8822–8878).

## Config delta vs olive-terrain-11

- **None** — identical config; this is the longer re-run after olive was killed at 3900.

## Config delta vs skilled-waterfall-10

- Code: **centered** `slot_diversity_loss` (commit `ffc33ed`)
- `horizon_k`: 4 (waterfall actual) → **12**
- `lambda_cov=0.0027` (active)

## W&B

- Run name: `copper-sky-12`
- Run id: `ejror834`
- State: killed at `_step=5650` (~2h24m); chat discussion references ~step 4300
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ejror834

## Parent

[investigation_003](../DESCRIPTION.md)
