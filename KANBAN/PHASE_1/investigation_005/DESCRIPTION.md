# Investigation 005 — Can we complete the 15k acceptance run?

**Status:** ACTIVE  
**Opened:** 2026-06 (after `cerulean-snow-13` validated winning config)  
**Closed:** —

## Question

Can Phase 1 Stage 1 run for **15,000 steps** on full SSv2 with the winning collapse config
and pass acceptance gates: stable training, non-collapsed `c_t`, `coarse_vs_copy_ratio < 1`,
rank trajectory acceptable?

## Why it matters

This is the operational definition of "Phase 1 Stage 1 done" before Phase 2 (fine flow).
Partial wins on shorter runs do not unlock the hierarchy work.

## Parent context

- Spec gates: [`AGENT_FILES/PHASES/PHASE_1.md`](../../AGENT_FILES/PHASES/PHASE_1.md) §12
- Config from: [investigation_003](investigation_003/) (`cerulean-snow-13`)
- Stability guards from: [investigation_001](investigation_001/)

## Runs

| Run | Role |
|---|---|
| [`elated-snowflake-15`](elated-snowflake-15/) | Full 15k attempt — grad-skip death spiral at 8500 |
| [`drawn-elevator-16`](drawn-elevator-16/) | Resume from ~7500, LR halved — high skip rate (failed) |
| [`royal-cherry-17`](royal-cherry-17/) | Resume from 7500 + AGC (active) |
