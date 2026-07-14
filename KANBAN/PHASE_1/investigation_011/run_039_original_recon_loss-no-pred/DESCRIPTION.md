# Run 039 - `original_recon_loss + no-pred`

**Current W&B run name:** `Investigation 11 · Original reconstruction · Present only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_011](../)  
**W&B:** `kttd1fib` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/kttd1fib  
**State:** `finished`  
**Created:** 2026-06-30T10:22:24Z  
**Last history step:** 14950  
**Runtime in W&B export:** 6.27h  
**Mode:** present-reconstruction-only  
**Reading-cycle verdict:** **Collapsed rep** - Present reconstruction may improve, but c_t is still weakly spread or video-independent.

## Research Role

Present-only original reconstruction run without geometry regularizers; isolates present branch collapse.

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
| `lambda_var` | 0 |
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 0 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | relative_mse |
| `present_recon_only` | true |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 4 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 2.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/inv011_present_recon_only |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_var` | 0.5 | 0 |
| `lambda_sigreg` | 5 | 0 |
| `lambda_recon_pred` | 0.05 | 0 |
| `recon_loss_mode` | cosine | relative_mse |
| `present_recon_only` | false | true |
| `predict_residual` | true | false |
| `lr_coarse_flow` | 1.000e-04 | 2.000e-04 |
| `checkpoint_dir` | /workspace/ckpt/inv011_cosine_residual | /workspace/ckpt/inv011_present_recon_only |

## Chronological Linkage

- Previous W&B run: Run 038 [`new_recon_loss`](../run_038_new_recon_loss/).
- Next W&B run: Run 040 [`inv011_fixed_position_decoder`](../run_040_inv011_fixed_position_decoder/).
- Parent investigation conclusion: Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Present Reconstruction Only** cycle. Do not apply copy-ratio gates to this run because the future-prediction branch is off.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3078183479/wandb_manifest.json` (0 bytes), `config.yaml` (5302 bytes), `output.log` (234433 bytes), `requirements.txt` (3821 bytes), `wandb-metadata.json` (1859 bytes), `wandb-summary.json` (1154 bytes) |
| Logged artifacts | `run-kttd1fib-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

**Command** (present-only autoencoder control, legacy relative_mse loss, no geometry regularizers):

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --present-recon-only \
  --recon-loss-mode relative_mse --lambda-recon 0.05 \
  --lambda-var 0 --lambda-sigreg 0 --lambda-cov 0 --lambda-slot 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_original_recon_nopred --log-every 50 --diag-every 500
```

**What it tested / config delta:** disables prediction entirely (`--present-recon-only`) and trains
only D(B(e_t)) -> e_t under the legacy relative_mse loss with every geometry regularizer off. The
purest question: can present reconstruction ALONE, with no geometry pressure, make c information-rich?
