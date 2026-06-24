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

Winning **config** is validated for pre-8500 behavior; **full 15k completion** is not.

**Three failure modes on the same resume trajectory (ckpt 7500):**

1. **elated** — skip spiral (threshold 50), frozen weights, illusory latent health.
2. **drawn** — halved LR delayed skip to 8550, same freeze pattern (85% skips).
3. **royal** — **fresh** 15k + AGC from step 0 (not resume). 0% skips; dodged elated's
   8400–8500 spike; **new cliff at 8600** → active rank/slot collapse (13.9→5.8).

AGC fixes optimizer freeze; it does **not** fix the underlying late-resume instability basin.
Next attempt needs **AGC + halved flow LR** at minimum, plus stronger abort signals on
`L_flow` / `agc_Fc_max_ratio`.

## Open

- Whether AGC + 1e-4 LR completes 15k without skip or collapse
- Whether fresh-from-init 15k with AGC avoids the 8500 basin entirely
- Code: pre-AGC grad logging, L_flow-based abort, optional LR backoff on AGC spike
