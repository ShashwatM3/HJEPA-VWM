# Run 013 - `cerulean-snow-13`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `4lo4j7qb` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4lo4j7qb  
**State:** `killed`  
**Created:** 2026-06-19T12:51:57Z  
**Last history step:** 6900  
**Runtime in W&B export:** 3.12h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Best early collapse-control configuration in investigation_003; still not a passing prediction run.

This run sits inside **collapse diagnostics, variance floor, horizon, and slot-loss variants**. The parent investigation question is: **Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 15000 |
| `stage1_steps` | 15000 |
| `horizon_k` | 12 |
| `frame_stride` | 2 |
| `lambda_var` | 0.5 |
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 0 |
| `sigreg_warmup_steps` | n/a |
| `lambda_recon` | 0 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | n/a |
| `decoder_blocks` | n/a |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 2.000e-04 |
| `lr_decoder` | n/a |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 50 |
| `agc_enabled` | n/a |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_var` | 0.1 | 0.5 |
| `lambda_cov` | 0.0027 | 0 |
| `lambda_slot` | 0.05 | 0 |

## Chronological Linkage

- Previous W&B run: Run 012 [`copper-sky-12`](../run_012_copper-sky-12/).
- Next W&B run: Run 014 [`jolly-forest-14`](../run_014_jolly-forest-14/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3001799398/wandb_manifest.json` (0 bytes), `artifact/3001799582/wandb_manifest.json` (0 bytes), `artifact/3001803462/wandb_manifest.json` (0 bytes), `artifact/3005906520/wandb_manifest.json` (0 bytes), `artifact/3006230534/wandb_manifest.json` (0 bytes), `artifact/3006360743/wandb_manifest.json` (0 bytes), `artifact/3008182028/wandb_manifest.json` (0 bytes), `artifact/3019302174/wandb_manifest.json` (0 bytes), ... 9 more |
| Logged artifacts | `run-4lo4j7qb-history:v0` (wandb-history), `run-4lo4j7qb-events:v10` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — cerulean-snow-13

## What this run tested

Hypothesis: **`lambda_var=0.5`** (strong variance floor) + **`horizon_k=12`**, no slot loss,
fixes collapse without Goodhart. BRIEF_V0_3 "Run 6".

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Config delta vs copper-sky-12

- `lambda_slot`: 0.05 → **0**
- `lambda_var`: 0.10 → **0.5**

## W&B

- Run name: `cerulean-snow-13`
- Run id: `4lo4j7qb`
- Runtime: ~3h 7m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4lo4j7qb

## Parent

[investigation_003](../DESCRIPTION.md) — also informs [investigation_005](../../investigation_005/)

## Local log

`logs/cerulean-snow-13/output.log` (`./output.log`; not present locally) (truncated ~6900 steps in workspace copy)
