# Run 006 - `exalted-lion-6`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `wv69n7n5` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/wv69n7n5  
**State:** `finished`  
**Created:** 2026-06-13T14:35:43Z  
**Last history step:** 450  
**Runtime in W&B export:** 14.3m  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Research Role

Early Phase 1 diagnostic run used to read collapse and baseline behavior.

This run sits inside **collapse diagnostics, variance floor, horizon, and slot-loss variants**. The parent investigation question is: **Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2_tiny |
| `steps` | 500 |
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
| `steps` | 30000 | 500 |
| `stage1_steps` | 30000 | 15000 |
| `lr_bottleneck` | 2.000e-04 | 1.000e-04 |
| `lr_coarse_flow` | 4.000e-04 | 2.000e-04 |
| `grad_clip` | 1 | 0.5 |
| `grad_skip_threshold` | n/a | 50 |

## Chronological Linkage

- Previous W&B run: Run 005 [`peachy-terrain-5`](../../investigation_001/run_005_peachy-terrain-5/).
- Next W&B run: Run 007 [`sleek-leaf-7`](../run_007_sleek-leaf-7/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2985184438/wandb_manifest.json` (0 bytes), `artifact/2985185913/wandb_manifest.json` (0 bytes), `artifact/2987462999/wandb_manifest.json` (0 bytes), `artifact/2990483129/wandb_manifest.json` (0 bytes), `artifact/2990955666/wandb_manifest.json` (0 bytes), `artifact/3000836428/wandb_manifest.json` (0 bytes), `artifact/3019302142/wandb_manifest.json` (0 bytes), `artifact/3020317883/wandb_manifest.json` (0 bytes), ... 7 more |
| Logged artifacts | `run-wv69n7n5-history:v0` (wandb-history), `run-wv69n7n5-events:v8` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — exalted-lion-6

## What this run tested

Plan Phase 04 **P1 instrumented diagnostic** (~500 steps on `ssv2_tiny`, ~13 min):
confirm the new collapse instrumentation works on the real pipeline at low cost, and
check whether the `c_t` collapse seen in [`peachy-terrain-5`](../../investigation_001/run_005_peachy-terrain-5/)
is an **init/early transient** or **persistent**. It doubled as an **init-knob
ablation**: this run was launched while bottleneck init was still **flag-configurable**.

## Command

```bash
python train.py --data ssv2_tiny --steps 500 --log-every 50 --diag-every 100
```

(`--lambda-cov`, `--lambda-slot`, `--lambda-var` all default / off.)

## Config delta (verified vs W&B config `wv69n7n5`)

- **Init was flag-based, NOT the later orthogonal bake.** W&B config shows
  `bottleneck_query_init="small_gaussian"`, `bottleneck_query_init_scale=0.1`,
  `bottleneck_zero_init_out_mlp=false`. This run **predates** commit `62b94dd`
  (06-13 18:40 UTC, "bake init fixes into the model; reserve flags for empirical
  knobs") — it ran at 14:35 UTC. So the earlier KANBAN claim that "orthogonal queries
  + zero-init `out_mlp` were baked in" is **wrong for this run**: exalted-lion-6 is
  precisely the run whose result *justified* baking a default and deleting the init
  flags (it proved init knobs are a non-lever).
- `lambda_var=0.10` (default), `horizon_k=4`, `lambda_cov`/`lambda_slot` off
- Post–Run-1 stability retune (`611f2cd`: halved LRs, `grad_clip=0.5`, skip guard)
- **Head-averaged** `c_attn_entropy` only — the per-head `c_attn_entropy_min` fix
  (`a96c0d6`, 06-13 18:17 UTC) landed **after** this run, so the entropy reading here
  is the flawed head-averaged metric (see OBSERVATIONS).

## W&B

- Run name: `exalted-lion-6`
- Run id: `wv69n7n5`
- State: finished; ~13 min; `ssv2_tiny`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/wv69n7n5

## Parent

[investigation_003](../DESCRIPTION.md) — chronologically first run in this investigation
