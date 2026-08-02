# Run 004 - `charmed-haze-4`

**Current W&B run name:** `Investigation 02 · Throughput · After frame fix`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_002](../)  
**W&B:** `gj8ypv0d` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/gj8ypv0d  
**State:** `finished`  
**Created:** 2026-06-10T00:21:43Z  
**Last history step:** 150  
**Runtime in W&B export:** 4.4m  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

Final tiny-data smoke before long/full-data experiments.

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

No tracked config-highlight difference from the previous W&B run; read this as a repeat/continuation unless the preserved notes say otherwise.

## Chronological Linkage

- Previous W&B run: Run 003 [`comfy-glade-3`](../run_003_comfy-glade-3/).
- Next W&B run: Run 005 [`peachy-terrain-5`](../../investigation_001/run_005_peachy-terrain-5/).
- Parent investigation conclusion: The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2955270112/wandb_manifest.json` (0 bytes), `artifact/2985184354/wandb_manifest.json` (0 bytes), `artifact/2987470026/wandb_manifest.json` (0 bytes), `artifact/2990479490/wandb_manifest.json` (0 bytes), `artifact/2990952183/wandb_manifest.json` (0 bytes), `artifact/3000836415/wandb_manifest.json` (0 bytes), `config.yaml` (3518 bytes), `output.log` (1174 bytes), ... 3 more |
| Logged artifacts | `run-gj8ypv0d-history:v0` (wandb-history), `run-gj8ypv0d-events:v4` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — charmed-haze-4

## What this run tested

200-step timed smoke **after** decode-only-needed-frames fix (`commit 5e78caa`).
Validated bit-identical inputs with improved throughput.

## Command

```bash
time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

## Config delta vs efficient-aardvark-2

- Selective frame decode in `data.py`
- `num_threads=1` kept for VP9 `.webm` reliability

## W&B

- Run name: `charmed-haze-4`
- Run id: `gj8ypv0d`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/gj8ypv0d

## Parent

[investigation_002](../DESCRIPTION.md)
