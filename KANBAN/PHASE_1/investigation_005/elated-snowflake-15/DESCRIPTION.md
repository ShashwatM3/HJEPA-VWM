# Run — elated-snowflake-15

## What this run tested

Full **15k** Phase 1 acceptance attempt with winning collapse config from
`cerulean-snow-13`: full SSv2, `horizon_k=12`, `lambda_var=0.5`, no slot loss.

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Config delta vs cerulean-snow-13

Same CLI flags (fresh run from init, not resume). Intended as continuation of the
validated config to completion. W&B config confirms it is **identical** to
cerulean-snow-13's experiment config: `horizon_k=12`, `lambda_var=0.5`, `lambda_slot=0`,
`lambda_cov=0`, `lr_coarse_flow=2e-4`, `lr_bottleneck=1e-4`, `grad_skip_threshold=50`,
`grad_clip=0.5`, no AGC.

## W&B

- Run name: `elated-snowflake-15`
- Run id: `jhodg49x` (confirmed via MCP — earlier chat noted it as unverified)
- State: **crashed** at `_step=13850` (target 15000); runtime ~5h25m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jhodg49x

## Parent

[investigation_005](../DESCRIPTION.md)
