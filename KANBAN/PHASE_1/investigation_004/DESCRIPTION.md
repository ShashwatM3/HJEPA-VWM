# Investigation 004 — Is VICReg-C needed on top of a strong variance floor?

**Status:** PAUSED  
**Opened:** 2026-06 (Plan Phase 04 / collapse work)  
**Paused:** 2026-06 (`cerulean-snow-13` made strong `lambda_var` sufficient for first milestone)

## Question

Does an off-diagonal **covariance penalty** (VICReg-C) lift `c_effective_rank` further
without Goodharting copy ratio — beyond what `lambda_var=0.5` alone achieves?

## Why it matters

If rank plateaus near ~13 while spec soft-targets >30, decorrelation may be needed.
If `lambda_var=0.5` is enough to reach gates, VICReg-C adds complexity and supervisor
sign-off (reverses v0.2 "no covariance initially").

## Parent context

- Branched from: [investigation_003](investigation_003/)
- Code: `losses.py` covariance_floor, `--lambda-cov` (default 0)
- Analysis: [`AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/ANALYSIS_AND_DECISIONS.md`](../../AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/ANALYSIS_AND_DECISIONS.md)

## Runs

**None executed** as primary A/B. `L_cov` logged on all runs at `lambda_cov=0` for calibration.

## Planned matrix (not run)

| Run | `lambda_cov` | Purpose |
|---|---|---|
| A | 0 | baseline on full SSv2 |
| B | calibrated | VICReg-C A/B |

Superseded in priority by `lambda_var` sweep that produced `cerulean-snow-13`.
