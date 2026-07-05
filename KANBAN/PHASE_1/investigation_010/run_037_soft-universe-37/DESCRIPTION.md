# Run 037 - `soft-universe-37`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_010](../)  
**W&B:** `2vbo6pbm` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2vbo6pbm  
**State:** `finished`  
**Created:** 2026-06-29T11:38:25Z  
**Last history step:** 14950  
**Runtime in W&B export:** 6.79h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Healthy rep, no predictor** - Representation health is not the limiting issue; F_c still does not beat copy.

## Research Role

Canonical clean residual result: healthy representation, no predictor.

This run sits inside **clean residual run with optimizer/regularization plumbing**. The parent investigation question is: **Can the residual recipe produce a healthy representation and a predictor that beats copy in a clean 15k run?**

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
| `lambda_sigreg` | 5 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0.05 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | true |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 4 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/inv010_residual_230096d |

## Difference From Previous W&B Run

No tracked config-highlight difference from the previous W&B run; read this as a repeat/continuation unless the preserved notes say otherwise.

## Chronological Linkage

- Previous W&B run: Run 036 [`upbeat-frog-36`](../run_036_upbeat-frog-36/).
- Next W&B run: Run 038 [`new_recon_loss`](../../investigation_011/run_038_new_recon_loss/).
- Parent investigation conclusion: Run 037 proved a key negative: c_t can be high-rank, video-specific, and stable while F_c still fails the copy gate. That is the canonical healthy-representation/no-predictor result.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3076056390/wandb_manifest.json` (0 bytes), `config.yaml` (5050 bytes), `output.log` (243343 bytes), `requirements.txt` (3821 bytes), `wandb-metadata.json` (1689 bytes), `wandb-summary.json` (1557 bytes) |
| Logged artifacts | `run-2vbo6pbm-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

**Command** (the clean 15k residual run with SIGReg warmup on the fixed plumbing):

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 --lambda-sigreg 5.0 --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 2000 --predict-residual \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv010_residual_230096d --log-every 50 --diag-every 500
```

**What it tested / config delta:** same residual recipe as inv009 run 035 (graceful-river-35)
but run cleanly to a full 15k on the fixed optimizer/regularization plumbing. The controlled
question: does the residual recipe, given a clean full run, produce a predictor that beats copy?
