# Observations — Investigation 005 (15k acceptance)

## Prior art: cerulean-snow-13

See [investigation_003/cerulean-snow-13](../investigation_003/cerulean-snow-13/OBSERVATIONS.md).
Healthy through at least step 6900 in local log; established config viability.

## elated-snowflake-15 — timeline

| Phase | Steps | What happened |
|---|---|---|
| Healthy | 4500–8000 | `grad_skipped=0`, `grad_norm` ~2–3, copy ratio **0.83–0.96**, rank ~13.7, std ~1.04 |
| Break | **8500** | `grad_norm` **~170** → `grad_skipped=1`, copy ratio **~5.7** |
| Frozen | 8500–13850 | **Every step skipped** (~5300 steps, no weight updates), copy ratio 3.5–5.7 |
| End | 13850 | Crash; `grad_norm` ~249, still skipped |

## Belief evolution

### Wrong (external post-hoc)

"LR still warming at 8500 caused spike" — **false** for this run. Schedule is 1.5k warmup +
cosine decay; at step 8500 `lr_mult ~0.48`, LR **falling** since step 1500.

"Adam poisoned at 8500" — **misleading** for sustained skips. Step 8500 was **skipped** (no
`optimizer.step()`). Sustained `grad_skipped=1` means **raw backward grad > 50 every step**
with frozen weights, not Adam vetoing normal grads.

### Supported

- **No learning after 8500** — skip guard prevents weight and EMA updates.
- Latent metrics looked OK post-break because they are **forward-pass only** — misleading.
- **Usable checkpoint is pre-spike (~6500–8000)**, not final step.
- Resume should **halve coarse flow LR** (`--lr-coarse-flow 1e-4`); keep `lambda_var=0.5`, `horizon_k=12`.

## W&B report

Run report created in chat: `elated-snowflake-15 — Run Report` on W&B (entity
`smahalanobis-uc-davis`, project `hjepa-vwm`).

## Current belief

Winning **config** is validated; **full 15k completion** is not. Next attempt is resume, not
from-scratch.

## Open

- Where rank plateaus if training completes 15k
- Whether lower flow LR avoids step-8500 class spikes
- Auto-abort on sustained `grad_skipped` (manual for now)
