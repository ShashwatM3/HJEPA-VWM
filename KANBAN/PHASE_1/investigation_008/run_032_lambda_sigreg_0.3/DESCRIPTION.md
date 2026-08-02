# Run 032 - `lambda_sigreg_0.3`

**Current W&B run name:** `Investigation 08 · Isotropy weight · 0.3`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_008](../)  
**W&B:** `x7z6e0ah` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/x7z6e0ah  
**State:** `crashed`  
**Created:** 2026-06-27T17:44:36Z  
**Last history step:** 13150  
**Runtime in W&B export:** 5.71h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

SIGReg lambda=0.3 run; tests a mild isotropy push.

This run sits inside **SIGReg ladder on top of reconstruction-capacity recipe**. The parent investigation question is: **Does SIGReg break the d_c utilization ceiling, and if it does, does prediction improve?**

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
| `lambda_sigreg` | 0.3 |
| `sigreg_warmup_steps` | n/a |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 4 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/sigreg0.3_D512x4_nc32 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_sigreg` | 1 | 0.3 |
| `checkpoint_dir` | /workspace/ckpt/sigreg1.0_D512x4_nc32 | /workspace/ckpt/sigreg0.3_D512x4_nc32 |

## Chronological Linkage

- Previous W&B run: Run 031 [`lambda_sigreg_1.0`](../run_031_lambda_sigreg_1.0/).
- Next W&B run: Run 033 [`lambda_sigreg_10.0`](../run_033_lambda_sigreg_10.0/).
- Parent investigation conclusion: SIGReg is a real rank lever, especially at high weights. However, higher rank alone made prediction worse or left copy unbeaten, so geometry alone was not enough.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3060448214/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (2223 bytes) |
| Logged artifacts | `run-x7z6e0ah-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

**Command** (one arm of the 4-wide `lambda_sigreg` sweep, eager-plant-22 512x4 background):

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 --lambda-sigreg 0.3 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/sigreg0.3_D512x4_nc32 --log-every 50 --diag-every 500
```

**What it tested / config delta:** lambda_sigreg=0.3 (mild push) is the only variable. The
gentlest arm of the ladder.
