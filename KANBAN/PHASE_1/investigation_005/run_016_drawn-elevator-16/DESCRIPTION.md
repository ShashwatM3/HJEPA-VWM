# Run 016 - `drawn-elevator-16`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_005](../)  
**W&B:** `0n5mx3qf` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0n5mx3qf  
**State:** `finished`  
**Created:** 2026-06-22T04:01:20Z  
**Last history step:** 14950  
**Runtime in W&B export:** 3.49h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

Resume attempt after the first 15k failure; tests checkpoint/restart behavior.

This run sits inside **15k acceptance attempts, resume behavior, AGC and skip control**. The parent investigation question is: **Can the best-known collapse-control recipe finish a 15k Phase 1 run and meet the prediction gates?**

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
| `lambda_recon` | 0 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | n/a |
| `decoder_blocks` | n/a |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | n/a |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 50 |
| `agc_enabled` | n/a |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lr_coarse_flow` | 2.000e-04 | 1.000e-04 |

## Chronological Linkage

- Previous W&B run: Run 015 [`elated-snowflake-15`](../run_015_elated-snowflake-15/).
- Next W&B run: Run 017 [`royal-cherry-17`](../run_017_royal-cherry-17/).
- Parent investigation conclusion: Longer training exposed that nonzero variance was not enough. The runs either became unstable or stayed far from the prediction gates; rank was still low or collapsed, and copy remained competitive.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3020313676/wandb_manifest.json` (0 bytes), `artifact/3020318192/wandb_manifest.json` (0 bytes), `artifact/3028581466/wandb_manifest.json` (0 bytes), `artifact/3028586641/wandb_manifest.json` (0 bytes), `config.yaml` (3818 bytes), `output.log` (50174 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1321 bytes), ... 1 more |
| Logged artifacts | `run-0n5mx3qf-history:v0` (wandb-history), `run-0n5mx3qf-events:v2` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — drawn-elevator-16

## What this run tested

**Resume** from `phase1_step7500.pt` after `elated-snowflake-15`'s grad-skip death
spiral, with the **coarse-flow LR halved** (2e-4 → 1e-4) to test whether slower `F_c`
updates avoid the late-run instability. Hypothesis: the ~8500 break was optimizer
instability, not the collapse config — so a gentler flow LR on resume should survive it.

## Command (confirmed — resume, lr_coarse_flow=1e-4)

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

W&B history starts at `_step≈7550` (no rows below) → **confirmed a resume**, not a fresh
run. W&B config confirms `lr_coarse_flow=1e-4` (the only non-AGC run with the halved
flow LR), `lambda_var=0.5`, `horizon_k=12`, `grad_skip_threshold=50`, **no AGC**.

## W&B

- Run name: `drawn-elevator-16`
- Run id: `0n5mx3qf`
- State: finished; `_step` 7550→14950; runtime ~3h29m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0n5mx3qf

## Parent

[investigation_005](../DESCRIPTION.md)
