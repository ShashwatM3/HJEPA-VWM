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

- Branched from: [investigation_003](../investigation_003/)
- Code: VICReg-C covariance penalty added flag-gated in `7b05aef`; `--lambda-cov` CLI in
  `578421b` (default 0). Term is the mean squared off-diagonal of the `c_t` feature
  covariance, pooled across batch×slots.
- Analysis: [`AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/ANALYSIS_AND_DECISIONS.md`](../../AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/ANALYSIS_AND_DECISIONS.md)

## Runs

| Run | `lambda_cov` (verified W&B) | Role |
|---|---|---|
| [`exalted-lion-6`](../investigation_003/exalted-lion-6/) | — (flag absent) | Tiny diagnostic baseline (cross-link) |
| **Run A** = [`sleek-leaf-7`](../investigation_003/sleek-leaf-7/) | **0** | Full SSv2; logs `L_cov` only for calibration; stopped @3500 — **executed** |
| Run B (isolated calibrated `lambda_cov`) | — | **Not executed** — slot-loss path then `lambda_var=0.5` superseded it |

**λ_cov reality across adjacent runs (verified via MCP):** `serene-cloud-8`,
`skilled-waterfall-10`, `olive-terrain-11`, `copper-sky-12` all ran `lambda_cov=0.0027`
— but always as a gentle **adjunct to slot loss**, never isolated. `cerulean-snow-13`
and `jolly-forest-14` ran `lambda_cov=0`. So VICReg-C was **never** tested as the
primary isolated lever. Dedicated A/B was deprioritized after the `lambda_var=0.5` win
in [investigation_003](../investigation_003/).
