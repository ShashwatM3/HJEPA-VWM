# Next Steps - run 014 `jolly-forest-14`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants.
- Parent investigation next direction: The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.
- Next chronological W&B run: Run 015 [`elated-snowflake-15`](../../investigation_005/run_015_elated-snowflake-15/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — jolly-forest-14

## Why

Winning-config repeat that reproduced cerulean's healthy trajectory but **crashed at
step 3900** (state: crashed), so it never reached 15k. [`cerulean-snow-13`](../run_013_cerulean-snow-13/)
holds the canonical "Run 6 win"; the next move was simply to relaunch the same config and
push it all the way to 15k for the acceptance gate.

## Spawned

**Next run:** [`elated-snowflake-15`](../../investigation_005/run_015_elated-snowflake-15/) —
full 15k acceptance attempt with winning config (see cerulean `NEXT_STEPS`).

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```
