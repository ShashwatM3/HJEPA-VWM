# Next steps — Investigation 005

**Status: ACTIVE** — 15k acceptance not yet confirmed.

## Run chain

[`cerulean-snow-13`](../investigation_003/cerulean-snow-13/) (config win)
→ [`elated-snowflake-15`](elated-snowflake-15/) (15k failed at grad-skip 8500)
→ [`drawn-elevator-16`](drawn-elevator-16/) (resume @7500, LR halved — 85% skips)
→ [`royal-cherry-17`](royal-cherry-17/) (resume @7500 + AGC — 0% skips, rank collapse @8600+)

## Immediate — next run after royal-cherry-17

See [`royal-cherry-17/NEXT_STEPS.md`](royal-cherry-17/NEXT_STEPS.md). Recommended:

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 \
  --log-every 50 \
  --diag-every 500
```

AGC remains on by default. **Do not** resume from royal-cherry checkpoints.

**Watch:** `L_flow`, `agc_Fc_max_ratio`, `grad_skipped`, `coarse_vs_copy_ratio`, `c_effective_rank`.

**Abort** if `grad_skipped` sustained OR `L_flow > 1.5` for 200 steps OR rank drops below 10.

## On success

- New run folder per KANBAN protocol
- **Close investigation 005**
- Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../AGENT_FILES/PHASES/PHASE_2.md)

## On failure

See [`royal-cherry-17/NEXT_STEPS.md`](royal-cherry-17/NEXT_STEPS.md) escalation ladder.
