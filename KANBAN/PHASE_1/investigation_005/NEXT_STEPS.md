# Next steps — Investigation 005

**Status: ACTIVE** — 15k acceptance not yet confirmed.

## Run chain

[`cerulean-snow-13`](../investigation_003/cerulean-snow-13/) (config win)
→ [`elated-snowflake-15`](elated-snowflake-15/) (15k failed at grad-skip 8500)
→ [`drawn-elevator-16`](drawn-elevator-16/) (resume @7500, LR halved — high skip rate)
→ **[`agc-resume-17`](agc-resume-17/)** (resume @7500 + AGC, code `1ae2e09`)

## Immediate — launch agc-resume-17

Follow [`AGENT_FILES/SETUPS/SETUP.md`](../../AGENT_FILES/SETUPS/SETUP.md) Path B, then:

```bash
tmux new -s train
cd /workspace/hierarchal-jepa-flow-world-model
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --log-every 50 \
  --diag-every 500
```

**Why:** AGC clips moderate `F_c` backward spikes (elated step-8450 class) instead of
freezing training; `grad_skip_threshold=150` is tail-only.

**Watch:** `grad_skipped`, `agc_Fc_any_clipped`, `instability_warn`, `coarse_vs_copy_ratio`,
`c_effective_rank`.

**Abort** if `grad_skipped=1` sustained (>10% over 500 steps).

## On success

- Update [`agc-resume-17/OBSERVATIONS.md`](agc-resume-17/OBSERVATIONS.md)
- **Close investigation 005**
- Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../AGENT_FILES/PHASES/PHASE_2.md)

## On failure

See [`agc-resume-17/NEXT_STEPS.md`](agc-resume-17/NEXT_STEPS.md).
