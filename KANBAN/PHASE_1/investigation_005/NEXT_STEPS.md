# Next steps — Investigation 005

**Status: ACTIVE** — 15k acceptance not yet confirmed.

## Run chain

[`cerulean-snow-13`](../investigation_003/cerulean-snow-13/) (config win)
→ [`elated-snowflake-15`](elated-snowflake-15/) (15k failed at grad-skip 8500)
→ [`drawn-elevator-16`](drawn-elevator-16/) (resume @7500, LR halved)

## Immediate

[`drawn-elevator-16`](drawn-elevator-16/) — verify outcome on W&B (`0n5mx3qf`).

If resume not yet launched or must be relaunched:

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

**Why:** `elated-snowflake-15` froze after step 8500; checkpoint 7500 is last healthy
state; halved coarse-flow LR addresses late-run spike, not collapse config.

**Watch:** `grad_skipped`, `coarse_vs_copy_ratio`, `c_effective_rank`.

**Abort** if `grad_skipped=1` sustained (>10% over 500 steps).

## On success

- Update [`drawn-elevator-16/OBSERVATIONS.md`](drawn-elevator-16/OBSERVATIONS.md)
- **Close investigation 005**
- Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../AGENT_FILES/PHASES/PHASE_2.md)

## On failure

See [`drawn-elevator-16/NEXT_STEPS.md`](drawn-elevator-16/NEXT_STEPS.md).
