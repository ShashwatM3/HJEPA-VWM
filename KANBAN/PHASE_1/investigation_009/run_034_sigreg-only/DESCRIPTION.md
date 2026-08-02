# Run 034 - `sigreg-only`

**Current W&B run name:** `Investigation 09 · Isotropy control · No reconstruction`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_009](../)  
**W&B:** `xz3nabr9` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/xz3nabr9  
**State:** `crashed`  
**Created:** 2026-06-28T03:34:48Z  
**Last history step:** 14050  
**Runtime in W&B export:** 6.16h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Residual/SIGReg run without reconstruction prediction; tests the zero-residual framing.

This run sits inside **residual prediction and zero-change baseline diagnosis**. The parent investigation question is: **If F_c predicts the future residual instead of the full future latent, does it finally beat the zero-change/copy baseline?**

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
| `lambda_sigreg` | 6 |
| `sigreg_warmup_steps` | n/a |
| `lambda_recon` | 0 |
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
| `checkpoint_dir` | /workspace/ckpt/inv009_run1_sigreg6_full |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_sigreg` | 10 | 6 |
| `lambda_recon` | 0.05 | 0 |
| `checkpoint_dir` | /workspace/ckpt/sigreg10.0_D512x4_nc32 | /workspace/ckpt/inv009_run1_sigreg6_full |

## Chronological Linkage

- Previous W&B run: Run 033 [`lambda_sigreg_10.0`](../../investigation_008/run_033_lambda_sigreg_10.0/).
- Next W&B run: Run 035 [`sigreg-recon-residual`](../run_035_sigreg-recon-residual/).
- Parent investigation conclusion: Residual prediction made c_t more dynamic and healthier, but F_c mostly tied the zero-residual baseline. The failure moved from representation collapse toward predictor learning.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3064582464/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1779 bytes) |
| Logged artifacts | `run-xz3nabr9-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# fine-meadow-34 — Run 1: SIGReg substrate (λ_sigreg=6, no recon, full-latent)

**Wave:** [investigation_009](../DESCRIPTION.md) · **Position:** 1 of 2 (the substrate arm)
**Pair:** [graceful-river-35](../run_035_sigreg-recon-residual/) (residual prediction arm)
**W&B:** `xz3nabr9` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/xz3nabr9
**Commit:** `bc77db6` · **Status:** COMPLETE (manually ended at step 14050, ~6h10m)

## Hypothesis

Isolate SIGReg as the sole rank lever with the reconstruction machinery removed (`λ_recon=0`,
`λ_recon_pred=0`) and full-latent prediction (`F_c` predicts `c_{t+k}`, not Δ). Two questions:
(1) does the inv008 "rank↑ ⟺ prediction↓" law reproduce **without any recon confound**, and (2) is a
strong-SIGReg `c` a clean substrate to build prediction on?

Registered predictions ([../OBSERVATIONS.md](../OBSERVATIONS.md) §2026-06-28): rank ~55 (R1-P1);
`coarse_vs_copy_ratio` > 1, ~6–7, worse than the no-SIGReg baseline (R1-P2); `L_flow` ~0.85–0.90,
`c_std_mean` ~0.90–0.93 with the var floor active, no collapse (R1-P3); `L_recon_*` meaningless
because the decoder is never trained (R1-P4).

## Config delta (vs the inv008 full-latent control)

`λ_sigreg`: **6.0**. `λ_var`: 0.5. `λ_recon`: **0**, `λ_recon_pred`: **0** (recon stripped).
`predict_residual`: **false**. Decoder 512×4, `n_c`=32, `k`=12, `lr_coarse_flow`=1e-4.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-sigreg 6.0 --lambda-var 0.5 \
  --lambda-recon 0 --lambda-recon-pred 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv009_run1_sigreg6_full \
  --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
