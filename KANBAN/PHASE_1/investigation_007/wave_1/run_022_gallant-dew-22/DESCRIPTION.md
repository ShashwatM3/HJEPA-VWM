# Run 022 - `gallant-dew-22`

**Current W&B run name:** `Investigation 07 · Decoder capacity · Wide`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_007](../../)  
**W&B:** `708jrel8` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/708jrel8  
**State:** `crashed`  
**Created:** 2026-06-26T23:37:43Z  
**Last history step:** 8850  
**Runtime in W&B export:** 3.97h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Decoder/reconstruction capacity wave member; contributes to the capacity-floor negative result.

This run sits inside **decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis**. The parent investigation question is: **What binds the reconstruction-capacity floor: decoder size, reconstruction weight, number of slots, or latent utilization?**

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
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 2 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/L0.05_D512x2_nc32 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `decoder_blocks` | 4 | 2 |
| `checkpoint_dir` | /workspace/ckpt/L0.05_D512x4_nc32 | /workspace/ckpt/L0.05_D512x2_nc32 |

## Chronological Linkage

- Previous W&B run: Run 021 [`eager-plant-22`](../run_021_eager-plant-22/).
- Next W&B run: Run 023 [`toasty-donkey-21`](../run_023_toasty-donkey-21/).
- Parent investigation conclusion: Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3052403520/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (2391 bytes) |
| Logged artifacts | `run-708jrel8-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# gallant-dew-22 — decoder axis, 512×2 (width)

**Wave:** [Wave 1](../DESCRIPTION.md) · **Position:** 4 of 5 (decoder ladder, width)
**Prev:** [light-universe-24](../run_024_light-universe-24/) (weight axis ends) · **Next:** [eager-plant-22](../run_021_eager-plant-22/) (decoder depth)
**W&B:** `708jrel8` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/708jrel8
**Status:** COMPLETE (last logged ~8.5k, floor plateaued)

## Hypothesis

First of the two decoder runs, holding `lambda_recon` at the baseline 0.05 and *widening* the
decoder (256 → 512 dim, ~3.4× params). Tests the **decoder-bound** hypothesis (the tech-lead's
intuition): that one thin `D` is too small to expand `c → e`, so a wider decoder reconstructs better
and the floor drops. Width is the most direct capacity increase, so it goes first.

## Config delta (vs baseline)

`decoder_dim`: **256 → 512** (blocks held at 2). `lambda_recon`=0.05, `n_c`=32,
`lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 512 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L0.05_D512x2_nc32 --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
