# Run — elated-snowflake-15

## What this run tested

Full **15k** Phase 1 acceptance attempt with winning collapse config from
`cerulean-snow-13`: full SSv2, `horizon_k=12`, `lambda_var=0.5`, no slot loss.

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Config delta vs cerulean-snow-13

Same CLI flags (fresh run from init, not resume). Intended as continuation of
validated config to completion.

## W&B

- Run name: `elated-snowflake-15`
- Run id: `jhodg49x` (from chat / report tooling)
- Runtime: ~5h 25m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jhodg49x

## Parent

[investigation_005](../DESCRIPTION.md)
