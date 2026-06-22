# Observations — elated-snowflake-15

## Outcome

**Did not complete 15k usefully.** Crashed ~step 13850 after **~5300 steps of zero learning**
post step 8500.

## Key numbers

| When | Step | Highlights |
|---|---|---|
| Best window | 4500–8000 | copy ratio **0.83–0.96**, rank ~13.7, std ~1.04, no skips |
| Break | 8500 | `grad_norm` **170**, `grad_skipped=1`, copy ratio **5.7** |
| Frozen | 8500–13850 | skip every step, copy ratio **3.5–5.7** |
| Final | 13850 | `grad_norm` ~249, copy ratio **3.47** |

Latent health post-8500: cosine ~0.17, rank ~13.7, std ~1.10 — **misleading** (forward-only).

## What worked

Pre-8500 behavior matched `cerulean-snow-13` expectations: collapse fixed, beats copy,
stable grads.

## What broke

Sudden pre-clip grad spike → skip guard engaged → weights frozen in a region where every
subsequent batch still produced huge grads → no recovery.

## Interpretation

**Config is not the failure mode; optimizer instability late in run is.** Final checkpoint
is garbage; resume from **phase1_step7500** (or 6500–8000 window) with lower flow LR.

## Report

W&B run report: `elated-snowflake-15 — Run Report` (structured narrative + charts).
