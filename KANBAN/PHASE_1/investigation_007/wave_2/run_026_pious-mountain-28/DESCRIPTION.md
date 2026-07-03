# Run 026 - `pious-mountain-28`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_007](../../)  
**W&B:** `7u5zkw6t` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/7u5zkw6t  
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
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 64 |
| `decoder_dim` | 256 |
| `decoder_blocks` | 2 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/L0.05_D256x2_nc64 |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `n_c` | 256 | 64 |
| `checkpoint_dir` | /workspace/ckpt/L0.05_D256x2_nc256 | /workspace/ckpt/L0.05_D256x2_nc64 |

## Chronological Linkage

- Previous W&B run: Run 025 [`earnest-dragon-25`](../run_025_earnest-dragon-25/).
- Next W&B run: Run 027 [`quiet-firebrand-25`](../run_027_quiet-firebrand-25/).
- Parent investigation conclusion: Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3055390381/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (2391 bytes) |
| Logged artifacts | `run-7u5zkw6t-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# pious-mountain-28 — latent axis, n_c = 64 (latent 2×)

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 1 of 5 (latent ladder, step 1 — the core experiment)
**Prev:** [Wave 1 / eager-plant-22](../../wave_1/run_021_eager-plant-22/) · **Next:** [classic-yogurt-29](../run_028_classic-yogurt-29/) (n_c=128)
**W&B:** `7u5zkw6t` (attempt-1, **DELETED** — dead) · new run name/id TBD on re-run
**Status:** 🔁 RE-RUNNING on **GPU 0** of the 4-GPU wave. Attempt 1 died at step 200 (whole-pod
death, no data). TIER 0 — a core latent bookend. Launch: [`../../GUIDE.md`](../../GUIDE.md) §3c.

## Hypothesis (the one Wave 1 left untested)

The first rung of the latent axis — the **only** binding-constraint candidate Wave 1 didn't
eliminate. Doubling the slot count (32 → 64) doubles `c`'s bandwidth (8,192 → 16,384 numbers, 128:1
→ 64:1). If the 0.585 floor is **latent-capacity-bound** (the project's lead hypothesis, "it's `c`,
not `D`"), this is where the floor should first move *with* a rising `c_effective_rank` and a falling
`coarse_vs_copy_ratio`.

**Wave-1 prior against it:** the under-used axis is `d_c` (per-slot dim, rank ~13/256), which adding
*slots* does not address — so the more likely outcome is floor flat or only cosmetically lower. This
run + [earnest-dragon-25](../run_025_earnest-dragon-25/) (n_c=256) are the two that decide it.

## Config delta (vs baseline)

`n_c`: **32 → 64** (architecture change — not resume-compatible). `lambda_recon`=0.05, decoder 256×2,
`lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 64 \
  --checkpoint-dir /workspace/ckpt/L0.05_D256x2_nc64 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md). Re-run plan → [NEXT_STEPS.md](NEXT_STEPS.md).
