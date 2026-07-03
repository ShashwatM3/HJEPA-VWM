# Run 031 - `lambda_sigreg_1.0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_008](../)  
**W&B:** `jk8kj7h7` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jk8kj7h7  
**State:** `crashed`  
**Created:** 2026-06-27T17:44:36Z  
**Last history step:** 13100  
**Runtime in W&B export:** 5.69h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

SIGReg lambda=1 run; tests scale-matched isotropy pressure.

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
| `lambda_sigreg` | 1 |
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
| `checkpoint_dir` | /workspace/ckpt/sigreg1.0_D512x4_nc32 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_sigreg` | 3 | 1 |
| `checkpoint_dir` | /workspace/ckpt/sigreg3.0_D512x4_nc32 | /workspace/ckpt/sigreg1.0_D512x4_nc32 |

## Chronological Linkage

- Previous W&B run: Run 030 [`lambda_sigreg_3.0`](../run_030_lambda_sigreg_3.0/).
- Next W&B run: Run 032 [`lambda_sigreg_0.3`](../run_032_lambda_sigreg_0.3/).
- Parent investigation conclusion: SIGReg is a real rank lever, especially at high weights. However, higher rank alone made prediction worse or left copy unbeaten, so geometry alone was not enough.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3060448458/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (2223 bytes) |
| Logged artifacts | `run-jk8kj7h7-history:v0` (wandb-history) |
| API/group fetch caveats | none |
