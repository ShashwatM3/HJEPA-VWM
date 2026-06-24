# Observations — drawn-elevator-16

**W&B:** [`0n5mx3qf`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0n5mx3qf) · state **finished** · runtime ~3h 29m · steps **7500–14950**

## Outcome

**Failed acceptance.** Reached 15k step count but **84.7% of logged steps skipped** (127/150).
Only ~23 optimizer updates post-resume. Grad-skip death spiral recurred from step **8550**.

## Key numbers

| When | Step | Highlights |
|---|---|---|
| Resume | 7500 | rank 13.57, copy 1.33, `grad_skipped=0` |
| Spike zone | 8500 | `grad_norm` 42.8, copy 4.35, still no skip |
| First skip | **8550** | `grad_norm` 62.5, `L_flow` 1.71 |
| Sustained skip | 8550–14950 | **127/150 rows skipped** |
| Final diag | 14500 | copy 3.33, rank **13.62**, std 1.10, cosine 0.17 |

## Interpretation

Halved coarse-flow LR (`1e-4`) **delayed** first skip by ~100 steps vs elated (8450) but did
**not** prevent the death spiral. Forward latents stayed deceptively healthy (rank ~13.6) —
same misleading pattern as elated post-break — because weights were frozen most of the time.

## vs royal-cherry-17

| | drawn | royal |
|---|---|---|
| AGC | off | on |
| `lr_coarse_flow` | 1e-4 | 2e-4 |
| Skip rate | 85% | **0%** |
| Final rank | 13.62 | **5.84** (collapsed) |
| Final copy ratio | 3.33 | **12.73** |

Drawn froze in a mediocre basin; royal kept learning into a worse one. Neither passes gates.
