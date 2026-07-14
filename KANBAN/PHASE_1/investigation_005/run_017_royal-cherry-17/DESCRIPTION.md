# Run 017 - `royal-cherry-17`

**Current W&B run name:** `Investigation 05 · Acceptance run · Adaptive clipping`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_005](../)  
**W&B:** `0xv4upvb` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0xv4upvb  
**State:** `killed`  
**Created:** 2026-06-24T12:03:39Z  
**Last history step:** 11350  
**Runtime in W&B export:** 4.87h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

AGC-hardened resume run; separates skip/instability control from representation success.

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
| `lr_coarse_flow` | 2.000e-04 |
| `lr_decoder` | n/a |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lr_coarse_flow` | 1.000e-04 | 2.000e-04 |
| `grad_skip_threshold` | 50 | 150 |
| `agc_enabled` | n/a | true |

## Chronological Linkage

- Previous W&B run: Run 016 [`drawn-elevator-16`](../run_016_drawn-elevator-16/).
- Next W&B run: Run 018 [`fanciful-lake-18`](../../investigation_006/run_018_fanciful-lake-18/).
- Parent investigation conclusion: Longer training exposed that nonzero variance was not enough. The runs either became unstable or stayed far from the prediction gates; rank was still low or collapsed, and copy remained competitive.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3033234783/wandb_manifest.json` (0 bytes), `config.yaml` (3975 bytes), `output.log` (131049 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1281 bytes), `wandb-summary.json` (1111 bytes) |
| Logged artifacts | `run-0xv4upvb-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — royal-cherry-17

## What this run tested

**Fresh full-SSv2 Phase 1 run from step 0** with **adaptive gradient clipping (AGC)**
enabled (commit `1ae2e09`). Same cerulean config as [`elated-snowflake-15`](../run_015_elated-snowflake-15/)
(`horizon_k=12`, `lambda_var=0.5`, peak LRs 1e-4 / 2e-4, seed 42) plus AGC knobs.

> **Note:** KANBAN originally described a resume from `phase1_step7500.pt`. W&B history
> shows steps **0–11350** with warmup `lr_mult` from 0 — this was a **fresh run**, not a
> resume. See `OBSERVATIONS.md` §Run identity.

## Hypothesis

Per-tensor AGC (λ_B=0.20, λ_Fc=0.10) clips moderate `F_c` backward spikes; post-AGC
`grad_skip_threshold=150` skips only tail catastrophes. Training should pass the
step-8500 break that killed elated and complete 15k with learning intact.

## Command (actual — fresh, per W&B)

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --log-every 50 \
  --diag-every 500
```

AGC on by default after `1ae2e09`. No `--resume` in logged config.

## Config (W&B `0xv4upvb`)

| Knob | Value |
|---|---|
| `agc_enabled` | True |
| `agc_lambda_bottleneck` | 0.20 |
| `agc_lambda_coarse_flow` | 0.10 |
| `agc_eps` | 1e-3 |
| `grad_skip_threshold` | 150 |
| `grad_clip` | 0.5 |
| `instability_warn` | grad > 30 ∧ L_flow > 1.0 |
| `lr_coarse_flow` | 2e-4 |
| `lr_bottleneck` | 1e-4 |
| `lambda_var` | 0.5 |
| `horizon_k` | 12 |
| `seed` | 42 |

## W&B

- Run name: `royal-cherry-17`
- Run id: `0xv4upvb`
- State: **killed** at step 11350 (target 15000)
- Runtime: ~4h 52m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0xv4upvb

## Parent

[investigation_005](../DESCRIPTION.md)
