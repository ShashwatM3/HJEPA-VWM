# Observations - investigation_005

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Cross-Run Synthesis

Longer training exposed that nonzero variance was not enough. The runs either became unstable or stayed far from the prediction gates; rank was still low or collapsed, and copy remained competitive.

## Run-by-Run Evidence

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 15 | [`elated-snowflake-15`](run_015_elated-snowflake-15/) | `jhodg49x` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=13.6879; c_cross_video_cosine=0.1655; c_std_mean=1.1022; coarse_vs_copy_ratio=3.4732; coarse_vs_batch_mean_ratio=1.0147 |
| 16 | [`drawn-elevator-16`](run_016_drawn-elevator-16/) | `0n5mx3qf` | `finished` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=13.621; c_cross_video_cosine=0.167; c_std_mean=1.099; coarse_vs_copy_ratio=3.3319; coarse_vs_batch_mean_ratio=1.3557 |
| 17 | [`royal-cherry-17`](run_017_royal-cherry-17/) | `0xv4upvb` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Low-rank rep | c_effective_rank=5.837; c_cross_video_cosine=0.3521; c_std_mean=0.9447; coarse_vs_copy_ratio=12.7266; coarse_vs_batch_mean_ratio=1.8137 |

## Pattern Across The Branch

Best copy ratio in this branch was run 016 at 3.3319; best batch-mean ratio was run 015 at 1.0147. None should be read as a full Phase 1 pass unless both gates pass together.

Verdict distribution: Invalid=2, Low-rank rep=1.

## What Changed The Research Direction

The next pivot added reconstruction anchors to test whether c_t lacked usable information rather than just variance.

## Original Notes Preserved

# Observations — Investigation 005 (15k acceptance)

## Prior art: cerulean-snow-13

See [investigation_003/cerulean-snow-13](../investigation_003/run_013_cerulean-snow-13/OBSERVATIONS.md).
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
