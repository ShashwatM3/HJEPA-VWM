# Run — jolly-forest-14

**Winning-config confirmation run** (var=0.5, k=12, no slot). Previously logged as
config-TBD; W&B has since resolved it.

## What this run tested

A repeat of the [`cerulean-snow-13`](../cerulean-snow-13/) winning config — the first
attempt to take `lambda_var=0.5` + `horizon_k=12` (no slot, no cov) toward the full 15k
acceptance run. It is the bridge between the cerulean breakthrough and the formal 15k
attempt [`elated-snowflake-15`](../../investigation_005/elated-snowflake-15/); it crashed
early (`_step=3900`).

## Command (reconstructed from W&B config `8bkeeuio`)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

Config confirmed: `horizon_k=12`, `lambda_var=0.5`, `lambda_slot=0`, `lambda_cov=0`,
`grad_skip_threshold=50`, `lr_coarse_flow=2e-4`. Identical experiment config to
cerulean-snow-13. Created 06-20 07:13 UTC.

## Config delta vs cerulean-snow-13

- **None** (same experiment config). A confirmation / continuation attempt, not a new lever.

## W&B

- Run name: `jolly-forest-14`
- Run id: `8bkeeuio`
- State: crashed at `_step=3900` (~1h32m)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8bkeeuio

## Parent

[investigation_003](../DESCRIPTION.md) (winning config) → bridges to
[investigation_005](../../investigation_005/) (the 15k acceptance arc)

## Mapping note

Earlier KANBAN marked this "config TBD / mapping uncertain." Resolved via MCP: it **is**
the winning `lambda_var=0.5` k=12 config — not an exploratory or off-config run.
