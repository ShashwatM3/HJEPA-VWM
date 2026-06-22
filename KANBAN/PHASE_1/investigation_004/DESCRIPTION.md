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

| Run | Role |
|---|---|
| [`exalted-lion-6`](../investigation_003/exalted-lion-6/) | Tiny diagnostic baseline (cross-link) |
| **Run A** = [`sleek-leaf-7`](../investigation_003/sleek-leaf-7/) | Full SSv2, `lambda_cov=0`, stopped @3500 — **executed** |
| Run B (calibrated `lambda_cov`) | **Not executed** — slot-loss path superseded |

`L_cov` logged on all runs at `lambda_cov=0` for calibration. Dedicated VICReg-C A/B
was deprioritized after `lambda_var=0.5` win in [investigation_003](../investigation_003/).
