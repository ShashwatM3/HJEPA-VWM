# Run 005 - `peachy-terrain-5`

**Current W&B run name:** `Investigation 01 · Long baseline · Original schedule`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_001](../)  
**W&B:** `1chv2608` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/1chv2608  
**State:** `failed`  
**Created:** 2026-06-10T00:42:46Z  
**Last history step:** 10950  
**Runtime in W&B export:** 3.89h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

First long baseline attempt; exposes late instability and non-passing prediction.

This run sits inside **first long Phase 1 baseline and late-instability discovery**. The parent investigation question is: **Can the first long Phase 1 baseline train far enough to produce a meaningful prediction signal?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2_tiny |
| `steps` | 30000 |
| `stage1_steps` | 30000 |
| `horizon_k` | 4 |
| `frame_stride` | 2 |
| `lambda_var` | 0.1 |
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
| `lr_bottleneck` | 2.000e-04 |
| `lr_coarse_flow` | 4.000e-04 |
| `lr_decoder` | n/a |
| `grad_clip` | 1 |
| `grad_skip_threshold` | n/a |
| `agc_enabled` | n/a |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `steps` | 200 | 30000 |

## Chronological Linkage

- Previous W&B run: Run 004 [`charmed-haze-4`](../../investigation_002/run_004_charmed-haze-4/).
- Next W&B run: Run 006 [`exalted-lion-6`](../../investigation_003/run_006_exalted-lion-6/).
- Parent investigation conclusion: The long baseline was not a success signal. It exposed late instability and weak prediction, which made optimizer hardening, restart discipline, and clearer acceptance metrics necessary before interpreting longer runs.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2985182449/wandb_manifest.json` (0 bytes), `artifact/2985184442/wandb_manifest.json` (0 bytes), `artifact/2987470025/wandb_manifest.json` (0 bytes), `artifact/2990481167/wandb_manifest.json` (0 bytes), `artifact/2990949217/wandb_manifest.json` (0 bytes), `artifact/3000836411/wandb_manifest.json` (0 bytes), `config.yaml` (3461 bytes), `output.log` (50893 bytes), ... 3 more |
| Logged artifacts | `run-1chv2608-history:v0` (wandb-history), `run-1chv2608-events:v4` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — peachy-terrain-5

## What this run tested

First end-to-end Phase 1 training launch on RunPod (`ssv2_tiny`, original
pre-Run-1 hyperparameters). Isolated whether the v0.2 stack trains at all.

## Command

```bash
python train.py --data ssv2_tiny --steps 30000
```

Command confirmed from the pod launch (`cursor_messages` ~1684/2133) and W&B config
(`max_steps=30000`). All hyperparameters came from `config.py` defaults at the
pre-Run-1 (commit `5cb8330`/`e1e1956`) era — there were no override flags yet beyond
`--data`/`--steps`.

Pre-retune defaults (verified vs W&B config `1chv2608`): `lr_coarse_flow=4e-4`,
`lr_bottleneck=2e-4`, `warmup_steps=10000`, `grad_clip=1.0`, **no `grad_skip_threshold`**
(the skip guard did not exist yet), `lambda_var=0.10`, `horizon_k=4`, `precision=bf16`,
`stage1_steps=30000`, `checkpoint_every=5000`. This is the **only** run in the project
with the 30k/10k-warmup schedule, `grad_clip=1.0`, and the un-halved LRs — every
later run uses the post-crash retune (`611f2cd`).

## Config delta

Baseline first launch — no prior run on this codebase.

## W&B

- Run name: `peachy-terrain-5`
- Run id: `1chv2608`
- State: **failed** at `_step≈10950` (NaN); runtime ~3h53m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/1chv2608

## Parent investigation

[investigation_001](../DESCRIPTION.md)
