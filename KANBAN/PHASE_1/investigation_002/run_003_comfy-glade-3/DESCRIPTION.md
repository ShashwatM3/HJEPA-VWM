# Run 003 - `comfy-glade-3`

**Current W&B run name:** `Investigation 02 · Dense logging · Before frame fix`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_002](../)  
**W&B:** `0mgmqxxi` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0mgmqxxi  
**State:** `finished`  
**Created:** 2026-06-09T02:52:33Z  
**Last history step:** 199  
**Runtime in W&B export:** 5.5m  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

Second short smoke to confirm repeatability after initial pod setup.

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

- Previous W&B run: Run 002 [`efficient-aardvark-2`](../run_002_efficient-aardvark-2/).
- Next W&B run: Run 004 [`charmed-haze-4`](../run_004_charmed-haze-4/).
- Parent investigation conclusion: The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2955294907/wandb_manifest.json` (0 bytes), `artifact/2985184335/wandb_manifest.json` (0 bytes), `config.yaml` (3518 bytes), `output.log` (35915 bytes), `requirements.txt` (3923 bytes), `wandb-metadata.json` (1192 bytes), `wandb-summary.json` (699 bytes) |
| Logged artifacts | `run-0mgmqxxi-history:v0` (wandb-history), `run-0mgmqxxi-events:v0` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — comfy-glade-3

## What this run tested

The **dense-logged** pre-fix smoke. After `efficient-aardvark-2` established the
~1.66 s/step baseline with the default `log_every=50`, this run re-ran the same
200-step `ssv2_tiny` smoke with **per-step logging** (`--log-every 1`) so every
step's `loss`/`L_flow`/`L_var`/`grad_norm`/`ema_m`/`lr_mult` was visible — the user
had asked why the loop only printed every 50 steps, which motivated wiring the
`--log-every` CLI flag (commit `4c1abb3`).

## Command

```bash
python train.py --data ssv2_tiny --steps 200 --log-every 1
```

W&B config confirms `log_every=1` (the only run in the project with that value),
`diag_every=500`, `max_steps=200`, `dataset=ssv2_tiny` — i.e. the proposed
`--diag-every 100` from the chat draft was **not** passed at launch.

## Config delta

Pre-fix dataloader (commit `4c1abb3` era, before selective decode `5e78caa`). Same
model/optimizer config as `efficient-aardvark-2`; only `log_every` differs.

## W&B

- Run name: `comfy-glade-3`
- Run id: `0mgmqxxi`
- Created: 2026-06-09 02:52 UTC (after `efficient-aardvark-2` 02:41, before `charmed-haze-4` 06-10)
- Runtime: ~5m 29s
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0mgmqxxi

## Parent

[investigation_002](../DESCRIPTION.md)
