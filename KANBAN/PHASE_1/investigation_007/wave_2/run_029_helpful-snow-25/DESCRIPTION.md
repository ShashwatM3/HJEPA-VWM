# Run 029 - `helpful-snow-25`

**Current W&B run name:** `Investigation 07 · Reconstruction weight · 1.00`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_007](../../)  
**W&B:** `tw685b5g` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/tw685b5g  
**State:** `crashed`  
**Created:** 2026-06-27T03:41:07Z  
**Last history step:** 200  
**Runtime in W&B export:** 5.8m  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

n_c/saturation wave member; early synchronized crash makes it operations evidence more than learning evidence.

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
| `lambda_recon` | 1 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 256 |
| `decoder_blocks` | 2 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/L1.0_D256x2_nc32 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_recon` | 0.05 | 1 |
| `n_c` | 128 | 32 |
| `checkpoint_dir` | /workspace/ckpt/L0.05_D256x2_nc128 | /workspace/ckpt/L1.0_D256x2_nc32 |

## Chronological Linkage

- Previous W&B run: Run 028 [`classic-yogurt-29`](../run_028_classic-yogurt-29/).
- Next W&B run: Run 030 [`lambda_sigreg_3.0`](../../../investigation_008/run_030_lambda_sigreg_3.0/).
- Parent investigation conclusion: Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3055396984/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (2389 bytes) |
| Logged artifacts | `run-tw685b5g-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# helpful-snow-25 — weight saturation, λ_recon = 1.0

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 4 of 5 (weight-axis saturation extreme)
**Prev:** [earnest-dragon-25](../run_025_earnest-dragon-25/) (n_c=256) · **Next:** [quiet-firebrand-25](../run_027_quiet-firebrand-25/) (combined)
**Extends:** the Wave-1 weight ladder ([light-universe-24](../../wave_1/run_024_light-universe-24/), λ=0.5)
**W&B:** `tw685b5g` (attempt-1, **DELETED** from W&B — dead link)
**Status:** ❌ DROPPED from the 4-GPU re-run. Attempt 1 died at step 200 (whole-pod death); the
re-run has only 4 GPUs, and this λ=1.0 weight-saturation run is the cut — see below / [Wave 2
DESCRIPTION](../DESCRIPTION.md).

## Hypothesis

The saturation extreme of the **weight** axis: λ_recon = 1.0 makes reconstruction *equal* to
`L_flow`'s weight (20× the baseline). Closes the OFAT blind spot for the weight ladder — confirms
whether the floor stays flat at the extreme or whether the faint Wave-1 trend suddenly bites. Also
the run where `L_flow` degradation (recon over-competing with prediction) should be most visible.

**Prediction (Wave 1):** floor ~0.581 (extrapolated from the −0.005/doubling slope), no break, with
`L_flow` ticking up. The most predictable run in the wave.

## Forensic significance

This run is the **architectural control** for the whole failed wave: it is identical in architecture
to Wave 1 (n_c=32, decoder 256×2) — only the weight differs. Because it **died at the same step 200
as the n_c=256 run**, it proves the Wave-2 failure was *not* an `n_c` shape bug but a whole-pod event.

## Config delta (vs baseline)

`lambda_recon`: **0.05 → 1.0** (20×). Decoder 256×2, `n_c`=32, `lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L1.0_D256x2_nc32 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md).
