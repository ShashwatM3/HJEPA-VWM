# Run 027 - `quiet-firebrand-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_007](../../)  
**W&B:** `bbrrydax` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/bbrrydax  
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
| `lambda_recon` | 0.2 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 64 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 2 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/L0.2_D512x2_nc64 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_recon` | 0.05 | 0.2 |
| `decoder_dim` | 256 | 512 |
| `checkpoint_dir` | /workspace/ckpt/L0.05_D256x2_nc64 | /workspace/ckpt/L0.2_D512x2_nc64 |

## Chronological Linkage

- Previous W&B run: Run 026 [`pious-mountain-28`](../run_026_pious-mountain-28/).
- Next W&B run: Run 028 [`classic-yogurt-29`](../run_028_classic-yogurt-29/).
- Parent investigation conclusion: Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3055561063/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (2389 bytes) |
| Logged artifacts | `run-bbrrydax-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# quiet-firebrand-25 — combined "all bigger" (λ=0.2 · 512×2 · n_c=64)

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 5 of 5 (combined interaction run — last of Wave 2)
**Prev:** [helpful-snow-25](../run_029_helpful-snow-25/) (weight saturation) · **Next:** — (end of the planned sweep)
**W&B:** `bbrrydax` (attempt-1, **DELETED** — dead) · new run name/id TBD on re-run
**Status:** 🔁 RE-RUNNING on **GPU 3** of the 4-GPU wave — **kept** as the wave's one
interaction-effect check (it took the slot freed by dropping λ=1.0). Attempt 1 died at step 200
(whole-pod death, no data). Launch: [`../../GUIDE.md`](../../GUIDE.md) §3c.

## Hypothesis

The only non-OFAT run in the sweep: it stacks a moderate dose of **all three** levers at once
(`lambda_recon`=0.2, decoder 512×2, `n_c`=64) to test whether they **compound** — i.e. whether the
floor drops more than any single axis alone. The "best shot at the floor" run.

**Prediction (Wave 1):** floor ~0.575–0.585, no break — because Wave 1 showed each axis is
independently flat and the limit is a *shared* mechanism (utilization of `d_c`), so the levers are
not expected to compound. Confounded by design (3 changes at once), so strictly less informative than
the clean single-axis runs.

## Config delta (vs baseline)

`lambda_recon` 0.05→**0.2**, `decoder_dim` 256→**512**, `n_c` 32→**64** (all at once; architecture
change). `lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.2 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 512 --decoder-blocks 2 --n-c 64 \
  --checkpoint-dir /workspace/ckpt/L0.2_D512x2_nc64 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md).
