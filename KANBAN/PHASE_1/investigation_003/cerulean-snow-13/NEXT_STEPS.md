# Next steps — cerulean-snow-13

## Why

Breakthrough: `lambda_var=0.5` + `horizon_k=12` fixed collapse without Goodhart —
std ~1.0, cosine ~0.20, rank climbing (~13.7+), beats copy, stable grads. Winning
config for Phase 1 Stage 1.

## Spawned

**Close** [investigation_003](../DESCRIPTION.md) with this config.

**Next investigation:** [investigation_005](../../investigation_005/) — operational
question: can this config **finish 15k** and pass acceptance gates?

**Next run:** [`elated-snowflake-15`](../../investigation_005/elated-snowflake-15/) —
same flags, fresh run from init (not resume), to completion.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

**Watch:** rank plateau, `coarse_vs_copy_ratio` hold, `grad_skipped`.

**Only if rank plateaus low after a stable full run:** [investigation_004](../../investigation_004/) (isolated VICReg-C A/B).

Do not re-enable slot loss.
