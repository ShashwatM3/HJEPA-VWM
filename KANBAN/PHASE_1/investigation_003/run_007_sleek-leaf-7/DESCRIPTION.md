# Run 007 - `sleek-leaf-7`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `rpxyg9qt` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/rpxyg9qt  
**State:** `crashed`  
**Created:** 2026-06-13T18:52:06Z  
**Last history step:** 4450  
**Runtime in W&B export:** 1.64h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

Longer collapse-control run that tested whether early health survives beyond smoke scale.

This run sits inside **collapse diagnostics, variance floor, horizon, and slot-loss variants**. The parent investigation question is: **Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 15000 |
| `stage1_steps` | 15000 |
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
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 2.000e-04 |
| `lr_decoder` | n/a |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 50 |
| `agc_enabled` | n/a |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `dataset` | ssv2_tiny | ssv2 |
| `steps` | 500 | 15000 |

## Chronological Linkage

- Previous W&B run: Run 006 [`exalted-lion-6`](../run_006_exalted-lion-6/).
- Next W&B run: Run 008 [`serene-cloud-8`](../run_008_serene-cloud-8/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2985183447/wandb_manifest.json` (0 bytes), `artifact/2985183878/wandb_manifest.json` (0 bytes), `artifact/2987466073/wandb_manifest.json` (0 bytes), `artifact/2990480993/wandb_manifest.json` (0 bytes), `artifact/2990949421/wandb_manifest.json` (0 bytes), `artifact/3000836413/wandb_manifest.json` (0 bytes), `artifact/3019302077/wandb_manifest.json` (0 bytes), `artifact/3020313695/wandb_manifest.json` (0 bytes), ... 5 more |
| Logged artifacts | `run-rpxyg9qt-history:v0` (wandb-history), `run-rpxyg9qt-events:v9` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — sleek-leaf-7

**BRIEF Run 2 / VICReg Run A.** First full-SSv2 baseline after P1 diagnostic.

## What this run tested

Init fixes + full SSv2, no slot/cov active (`lambda_cov=0` logs `L_cov` only).
Isolated: does init + full data fix rank ~5? Stopped ~step 3500 for data-confound verdict.

## Command (confirmed — chat `cursor_messages` ~5832/5867; W&B config `rpxyg9qt`)

```bash
python train.py --data ssv2 --steps 15000 --log-every 50 --diag-every 500
```

Post-retune defaults; init fixes baked in (`62b94dd`); per-head entropy fix (`a96c0d6`).
W&B config confirms `lambda_var=0.10`, `horizon_k=4`, `lambda_cov=0`, `lambda_slot` unset
(slot loss `4418410` had not landed yet — this is a clean no-regularizer baseline). Run
stopped ~step 3500 for the data-confound verdict (W&B shows it reached `_step=4450`).

## Config delta vs peachy-terrain-5

- Full SSv2 (not tiny)
- Halved LRs, 1.5k warmup, 15k steps, grad clip 0.5, skip guard
- Per-head `c_attn_entropy_min` active (post `a96c0d6`)

## W&B

- Run name: `sleek-leaf-7`
- Run id: `rpxyg9qt`
- Runtime: ~1h 39m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/rpxyg9qt

## Parent

[investigation_003](../DESCRIPTION.md) — also [investigation_004 Run A](../../investigation_004/)
