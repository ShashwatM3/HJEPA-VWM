# Run 049 - `po_geom_sig10_cov0p01`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_011](../../../)  
**W&B:** `bttexglp` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/bttexglp  
**State:** `crashed`  
**Created:** 2026-07-02T03:48:46Z  
**Last history step:** 14350  
**Runtime in W&B export:** 5.70h  
**Mode:** present-reconstruction-only  
**Reading-cycle verdict:** **Strong present representation** - The present branch is decodable, spread, video-specific, and high-rank.

## Research Role

Present-only geometry sweep wave 2 member: sigreg=10, cov=0.01.

This run sits inside **cosine reconstruction, fixed-position decoder, present-only geometry sweeps**. The parent investigation question is: **Can reconstruction geometry produce a strong, decodable present bottleneck, and can that evidence be separated from prediction failure?**

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
| `lambda_cov` | 0.01 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 10 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | cosine |
| `present_recon_only` | true |
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
| `checkpoint_dir` | /workspace/ckpt/inv011_present_only_geometry_sweep/po_geom_sig10_cov0p01 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_cov` | 0.003 | 0.01 |
| `lambda_sigreg` | 7.5 | 10 |
| `checkpoint_dir` | /workspace/ckpt/inv011_present_only_geometry_sweep/po_geom_sig7p5_cov0p003 | /workspace/ckpt/inv011_present_only_geometry_sweep/po_geom_sig10_cov0p01 |

## Chronological Linkage

- Previous W&B run: Run 048 [`po_geom_sig7p5_cov0p003`](../run_048_po_geom_sig7p5_cov0p003/).
- Next W&B run: Run 050 [`po_geom_sig7p5_cov0p01`](../run_050_po_geom_sig7p5_cov0p01/).
- Parent investigation conclusion: Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Present Reconstruction Only** cycle. Do not apply copy-ratio gates to this run because the future-prediction branch is off.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3087332880/wandb_manifest.json` (0 bytes), `requirements.txt` (3821 bytes), `wandb-metadata.json` (2638 bytes) |
| Logged artifacts | `run-bttexglp-history:v0` (wandb-history) |
| API/group fetch caveats | none |
