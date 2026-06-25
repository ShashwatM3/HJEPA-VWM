# Next steps — jolly-forest-14

## Why

Winning-config repeat that reproduced cerulean's healthy trajectory but **crashed at
step 3900** (state: crashed), so it never reached 15k. [`cerulean-snow-13`](../cerulean-snow-13/)
holds the canonical "Run 6 win"; the next move was simply to relaunch the same config and
push it all the way to 15k for the acceptance gate.

## Spawned

**Next run:** [`elated-snowflake-15`](../../investigation_005/elated-snowflake-15/) —
full 15k acceptance attempt with winning config (see cerulean `NEXT_STEPS`).

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```
