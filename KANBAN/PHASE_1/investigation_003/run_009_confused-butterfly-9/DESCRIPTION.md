# Run 009 - `confused-butterfly-9`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `m30jxiye` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/m30jxiye  
**State:** `killed`  
**Created:** 2026-06-16T18:15:37Z  
**Last history step:** -1  
**Runtime in W&B export:** 0s  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

Failed/early-abort diagnostic entry; records launch/state failure rather than learning.

This run sits inside **collapse diagnostics, variance floor, horizon, and slot-loss variants**. The parent investigation question is: **Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 5000 |
| `stage1_steps` | 15000 |
| `horizon_k` | 4 |
| `frame_stride` | 2 |
| `lambda_var` | 0.1 |
| `lambda_cov` | 0.0027 |
| `lambda_slot` | 0.25 |
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

- Previous W&B run: Run 008 [`serene-cloud-8`](../run_008_serene-cloud-8/).
- Next W&B run: Run 010 [`skilled-waterfall-10`](../run_010_skilled-waterfall-10/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `config.yaml` (3771 bytes), `output.log` (881 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1287 bytes), `wandb-summary.json` (37 bytes) |
| Logged artifacts | none listed in export |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — confused-butterfly-9

## What this run tested

Nothing useful — **immediate failure** (0s compute, no metrics logged). A mis-launch of
the slot-loss experiment: W&B config matches the planned **k=4, slot=0.25** slot run
(`horizon_k=4`, `lambda_slot=0.25`, `lambda_cov=0.0027`, `max_steps=5000`,
`diag_every=250`). It was created 06-16 18:15 UTC, **one second before**
[`skilled-waterfall-10`](../run_010_skilled-waterfall-10/) (18:16) — i.e. a fat-finger / instant
re-launch, not a distinct experiment.

## W&B

- Run name: `confused-butterfly-9`
- Run id: `m30jxiye`
- State: killed; **0s** compute; only `_runtime:0` logged
- Config: k=4, λ_slot=0.25, λ_cov=0.0027, 5000 steps (the serene-cloud-8 slot config)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/m30jxiye

## Parent

[investigation_003](../DESCRIPTION.md)

## Note

Config recovered from W&B (no chat command preserved). Discard for metrics — it produced
none. Its only informational value is dating the slot-run launch sequence.
