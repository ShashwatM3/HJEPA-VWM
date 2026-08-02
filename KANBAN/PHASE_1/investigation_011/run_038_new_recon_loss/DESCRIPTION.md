# Run 038 - `new_recon_loss`

**Current W&B run name:** `Investigation 11 · Cosine reconstruction · Full prediction`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_011](../)  
**W&B:** `1u69hpfm` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/1u69hpfm  
**State:** `finished`  
**Created:** 2026-06-30T10:17:00Z  
**Last history step:** 14950  
**Runtime in W&B export:** 6.36h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Healthy rep, no predictor** - Representation health is not the limiting issue; F_c still does not beat copy.

## Research Role

Cosine reconstruction full-prediction run; tests improved readout geometry with residual prediction.

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
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 5 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0.05 |
| `recon_loss_mode` | cosine |
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
| `checkpoint_dir` | /workspace/ckpt/inv011_cosine_residual |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `recon_loss_mode` | n/a | cosine |
| `checkpoint_dir` | /workspace/ckpt/inv010_residual_230096d | /workspace/ckpt/inv011_cosine_residual |

## Chronological Linkage

- Previous W&B run: Run 037 [`soft-universe-37`](../../investigation_010/run_037_soft-universe-37/).
- Next W&B run: Run 039 [`original_recon_loss + no-pred`](../run_039_original_recon_loss-no-pred/).
- Parent investigation conclusion: Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3078561189/wandb_manifest.json` (0 bytes), `config.yaml` (5499 bytes), `output.log` (259693 bytes), `requirements.txt` (3821 bytes), `wandb-metadata.json` (1964 bytes), `wandb-summary.json` (1604 bytes) |
| Logged artifacts | `run-1u69hpfm-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

**Command** (run 037's exact residual recipe with the reconstruction formula switched to cosine):

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 --lambda-sigreg 5.0 --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-loss-mode cosine --recon-warmup-steps 2000 \
  --predict-residual --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_new_recon_loss --log-every 50 --diag-every 500
```

**What it tested / config delta:** the only change vs run 037 is `--recon-loss-mode cosine`
(norm-invariant per-tubelet cosine distance, replacing the legacy MSE/Var which let the decoder
game feature norms). Tests whether the improved recon geometry helps the best full-prediction recipe.
