# Next Steps - run 013 `cerulean-snow-13`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants.
- Parent investigation next direction: The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.
- Next chronological W&B run: Run 014 [`jolly-forest-14`](../run_014_jolly-forest-14/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — cerulean-snow-13

## Why

Breakthrough: `lambda_var=0.5` + `horizon_k=12` fixed collapse without Goodhart —
std ~1.0, cosine ~0.20, rank climbing (~13.7+), beats copy, stable grads. Winning
config for Phase 1 Stage 1.

## Spawned

**Close** [investigation_003](../DESCRIPTION.md) with this config.

**Next investigation:** [investigation_005](../../investigation_005/) — operational
question: can this config **finish 15k** and pass acceptance gates?

**Next run:** [`elated-snowflake-15`](../../investigation_005/run_015_elated-snowflake-15/) —
same flags, fresh run from init (not resume), to completion.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

**Watch:** rank plateau, `coarse_vs_copy_ratio` hold, `grad_skipped`.

**Only if rank plateaus low after a stable full run:** [investigation_004](../../investigation_004/) (isolated VICReg-C A/B).

Do not re-enable slot loss.
