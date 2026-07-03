# Run 002 - `efficient-aardvark-2`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_002](../)  
**W&B:** `fz7ztfc8` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fz7ztfc8  
**State:** `finished`  
**Created:** 2026-06-09T02:41:14Z  
**Last history step:** 150  
**Runtime in W&B export:** 5.2m  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

Timed tiny-data smoke continuation; checks throughput and short-window stability.

This run sits inside **RunPod, dataloader, W&B, and tiny-data smoke validation**. The parent investigation question is: **Can the training stack launch, log, decode data, and produce interpretable metrics before expensive full-data experiments?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2_tiny |
| `steps` | 200 |
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

| Config key | Previous run | This run |
|---|---:|---:|
| `steps` | 100 | 200 |

## Chronological Linkage

- Previous W&B run: Run 001 [`youthful-pond-1`](../run_001_youthful-pond-1/).
- Next W&B run: Run 003 [`comfy-glade-3`](../run_003_comfy-glade-3/).
- Parent investigation conclusion: The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2955292530/wandb_manifest.json` (0 bytes), `artifact/2985170270/wandb_manifest.json` (0 bytes), `config.yaml` (3459 bytes), `output.log` (1173 bytes), `requirements.txt` (3923 bytes), `wandb-metadata.json` (1164 bytes), `wandb-summary.json` (699 bytes) |
| Logged artifacts | `run-fz7ztfc8-history:v0` (wandb-history), `run-fz7ztfc8-events:v0` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — efficient-aardvark-2

## What this run tested

200-step timed baseline **before** decode-only-needed-frames fix. Established
~1.66 s/step and confirmed metrics matched prior local smoke (`logs/1.json`).

## Command

```bash
time python train.py --data ssv2_tiny --steps 200
```

## Config delta

Pre-fix dataloader (decord decoded all frames then sliced). Commit `4c1abb3` era.

## W&B

- Run name: `efficient-aardvark-2`
- Run id: `fz7ztfc8`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fz7ztfc8

## Parent

[investigation_002](../DESCRIPTION.md)
