# Run 001 - `youthful-pond-1`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_002](../)  
**W&B:** `x4pwz33d` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/x4pwz33d  
**State:** `finished`  
**Created:** 2026-06-09T02:34:34Z  
**Last history step:** 50  
**Runtime in W&B export:** 2.4m  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

First W&B/data smoke run; validates launch, logging, and one diagnostic row on ssv2_tiny.

This run sits inside **RunPod, dataloader, W&B, and tiny-data smoke validation**. The parent investigation question is: **Can the training stack launch, log, decode data, and produce interpretable metrics before expensive full-data experiments?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2_tiny |
| `steps` | 100 |
| `stage1_steps` | 30000 |
| `horizon_k` | 4 |
| `frame_stride` | 2 |
| `lambda_var` | 0.1 |
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
| `lr_bottleneck` | 2.000e-04 |
| `lr_coarse_flow` | 4.000e-04 |
| `lr_decoder` | n/a |
| `grad_clip` | 1 |
| `grad_skip_threshold` | n/a |
| `agc_enabled` | n/a |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

This is the first W&B run in the project sequence.

## Chronological Linkage

- Previous W&B run: None; this is the first W&B run.
- Next W&B run: Run 002 [`efficient-aardvark-2`](../run_002_efficient-aardvark-2/).
- Parent investigation conclusion: The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2955329102/wandb_manifest.json` (0 bytes), `artifact/2985185135/wandb_manifest.json` (0 bytes), `artifact/2987468984/wandb_manifest.json` (0 bytes), `artifact/2990481164/wandb_manifest.json` (0 bytes), `artifact/2990934541/wandb_manifest.json` (0 bytes), `config.yaml` (3459 bytes), `output.log` (819 bytes), `requirements.txt` (3923 bytes), ... 2 more |
| Logged artifacts | `run-x4pwz33d-history:v0` (wandb-history), `run-x4pwz33d-events:v3` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — youthful-pond-1

## What this run tested

First end-to-end training smoke on RunPod after initial setup: does the v0.2 stack
launch, log to W&B, and produce sane step-0 metrics on `ssv2_tiny`?

## Command

```bash
python train.py --data ssv2_tiny --steps 100
```

## Config delta

Baseline first launch — no prior run on this codebase.

## W&B

- Run name: `youthful-pond-1`
- Run id: `x4pwz33d`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/x4pwz33d

## Parent

[investigation_002](../DESCRIPTION.md)
