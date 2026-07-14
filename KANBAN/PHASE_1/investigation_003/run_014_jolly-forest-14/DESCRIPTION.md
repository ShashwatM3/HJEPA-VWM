# Run 014 - `jolly-forest-14`

**Current W&B run name:** `Investigation 03 · Strong variance · Repeat`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `8bkeeuio` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8bkeeuio  
**State:** `crashed`  
**Created:** 2026-06-20T07:13:55Z  
**Last history step:** 3900  
**Runtime in W&B export:** 1.52h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Repeat of the best early configuration; crashed before completion and did not settle the gate.

This run sits inside **collapse diagnostics, variance floor, horizon, and slot-loss variants**. The parent investigation question is: **Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?**

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

- Previous W&B run: Run 013 [`cerulean-snow-13`](../run_013_cerulean-snow-13/).
- Next W&B run: Run 015 [`elated-snowflake-15`](../../investigation_005/run_015_elated-snowflake-15/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3008180608/wandb_manifest.json` (0 bytes), `artifact/3019306548/wandb_manifest.json` (0 bytes), `artifact/3020316930/wandb_manifest.json` (0 bytes), `artifact/3028581470/wandb_manifest.json` (0 bytes), `artifact/3028586655/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1220 bytes) |
| Logged artifacts | `run-8bkeeuio-history:v0` (wandb-history), `run-8bkeeuio-events:v3` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — jolly-forest-14

**Winning-config confirmation run** (var=0.5, k=12, no slot). Previously logged as
config-TBD; W&B has since resolved it.

## What this run tested

A repeat of the [`cerulean-snow-13`](../run_013_cerulean-snow-13/) winning config — the first
attempt to take `lambda_var=0.5` + `horizon_k=12` (no slot, no cov) toward the full 15k
acceptance run. It is the bridge between the cerulean breakthrough and the formal 15k
attempt [`elated-snowflake-15`](../../investigation_005/run_015_elated-snowflake-15/); it crashed
early (`_step=3900`).

## Command (reconstructed from W&B config `8bkeeuio`)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

Config confirmed: `horizon_k=12`, `lambda_var=0.5`, `lambda_slot=0`, `lambda_cov=0`,
`grad_skip_threshold=50`, `lr_coarse_flow=2e-4`. Identical experiment config to
cerulean-snow-13. Created 06-20 07:13 UTC.

## Config delta vs cerulean-snow-13

- **None** (same experiment config). A confirmation / continuation attempt, not a new lever.

## W&B

- Run name: `jolly-forest-14`
- Run id: `8bkeeuio`
- State: crashed at `_step=3900` (~1h32m)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8bkeeuio

## Parent

[investigation_003](../DESCRIPTION.md) (winning config) → bridges to
[investigation_005](../../investigation_005/) (the 15k acceptance arc)

## Mapping note

Earlier KANBAN marked this "config TBD / mapping uncertain." Resolved via MCP: it **is**
the winning `lambda_var=0.5` k=12 config — not an exploratory or off-config run.
