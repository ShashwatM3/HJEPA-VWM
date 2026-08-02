# Run 015 - `elated-snowflake-15`

**Current W&B run name:** `Investigation 05 · Acceptance run · Before adaptive clipping`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_005](../)  
**W&B:** `jhodg49x` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jhodg49x  
**State:** `crashed`  
**Created:** 2026-06-20T15:23:36Z  
**Last history step:** 13850  
**Runtime in W&B export:** 5.42h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

15k acceptance-style attempt; tests whether the best-known recipe finishes cleanly.

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
| `grad_skip_threshold` | 50 |
| `agc_enabled` | n/a |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

No tracked config-highlight difference from the previous W&B run; read this as a repeat/continuation unless the preserved notes say otherwise.

## Chronological Linkage

- Previous W&B run: Run 014 [`jolly-forest-14`](../../investigation_003/run_014_jolly-forest-14/).
- Next W&B run: Run 016 [`drawn-elevator-16`](../run_016_drawn-elevator-16/).
- Parent investigation conclusion: Longer training exposed that nonzero variance was not enough. The runs either became unstable or stayed far from the prediction gates; rank was still low or collapsed, and copy remained competitive.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3010010550/wandb_manifest.json` (0 bytes), `artifact/3010014579/wandb_manifest.json` (0 bytes), `artifact/3019302089/wandb_manifest.json` (0 bytes), `artifact/3020318271/wandb_manifest.json` (0 bytes), `artifact/3028581471/wandb_manifest.json` (0 bytes), `artifact/3028586647/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1220 bytes) |
| Logged artifacts | `run-jhodg49x-history:v0` (wandb-history), `run-jhodg49x-events:v4` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — elated-snowflake-15

## What this run tested

Full **15k** Phase 1 acceptance attempt with winning collapse config from
`cerulean-snow-13`: full SSv2, `horizon_k=12`, `lambda_var=0.5`, no slot loss.

## Command

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Config delta vs cerulean-snow-13

Same CLI flags (fresh run from init, not resume). Intended as continuation of the
validated config to completion. W&B config confirms it is **identical** to
cerulean-snow-13's experiment config: `horizon_k=12`, `lambda_var=0.5`, `lambda_slot=0`,
`lambda_cov=0`, `lr_coarse_flow=2e-4`, `lr_bottleneck=1e-4`, `grad_skip_threshold=50`,
`grad_clip=0.5`, no AGC.

## W&B

- Run name: `elated-snowflake-15`
- Run id: `jhodg49x` (confirmed via MCP — earlier chat noted it as unverified)
- State: **crashed** at `_step=13850` (target 15000); runtime ~5h25m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jhodg49x

## Parent

[investigation_005](../DESCRIPTION.md)
