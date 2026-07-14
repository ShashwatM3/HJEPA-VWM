# Run 010 - `skilled-waterfall-10`

**Current W&B run name:** `Investigation 03 · Mild slot loss · Long horizon`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `27i1r9qi` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/27i1r9qi  
**State:** `crashed`  
**Created:** 2026-06-16T18:16:19Z  
**Last history step:** 2550  
**Runtime in W&B export:** 1.14h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

k=4/raw-slot variant; important because it confirmed a launch/config drift and inert slot loss.

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
| `lambda_var` | 0.1 |
| `lambda_cov` | 0.0027 |
| `lambda_slot` | 0.05 |
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
| `steps` | 5000 | 15000 |
| `horizon_k` | 4 | 12 |
| `lambda_slot` | 0.25 | 0.05 |

## Chronological Linkage

- Previous W&B run: Run 009 [`confused-butterfly-9`](../run_009_confused-butterfly-9/).
- Next W&B run: Run 011 [`olive-terrain-11`](../run_011_olive-terrain-11/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2990479252/wandb_manifest.json` (0 bytes), `artifact/2990483542/wandb_manifest.json` (0 bytes), `artifact/2990949543/wandb_manifest.json` (0 bytes), `artifact/3000836410/wandb_manifest.json` (0 bytes), `artifact/3019302156/wandb_manifest.json` (0 bytes), `artifact/3020316186/wandb_manifest.json` (0 bytes), `artifact/3028581485/wandb_manifest.json` (0 bytes), `artifact/3028586656/wandb_manifest.json` (0 bytes), ... 2 more |
| Logged artifacts | `run-27i1r9qi-history:v0` (wandb-history), `run-27i1r9qi-events:v6` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — skilled-waterfall-10

**BRIEF "Run 4."** Intended as the *harder-horizon + mild-slot* run — but it executed
at **k=4**, and it ran on the **pre-centering (raw)** slot loss, so it is **not** a
clean test of either lever. Its lasting value was exposing the slot loss/metric
mismatch.

## What this run tested (intent vs reality)

**Intent:** harder horizon (k=12) + gentle slot loss (0.05) — does a harder prediction
task plus a mild slot penalty help, at a weight well below serene's Goodharting 0.25?

**Reality (W&B config `27i1r9qi`): `horizon_k=4`, not 12.** Despite the proposed
command below carrying `--horizon-k 12`, the launched config records `horizon_k=4`.
The `--horizon-k` flag (`183fbc8`) landed only ~4 min before this run started
(06-16 18:12 UTC vs run 18:16 UTC), so the flag was either not pulled or not passed —
a proposed-vs-actual command drift. **Treat this as a k=4 run.**

## Command (proposed — actual run was k=4)

```bash
# proposed:
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
# actual config: horizon_k=4, lambda_slot=0.05, lambda_cov=0.0027, lambda_var=0.10
```

## Config delta vs serene-cloud-8

- `lambda_slot`: 0.25 → **0.05**
- `horizon_k`: **stayed 4** (intended 12; see above)
- `lambda_cov=0.0027` (active), `lambda_var=0.10` (default)
- Slot loss is still the **raw** (pre-centering) formulation — `ffc33ed` had not landed

## W&B

- Run name: `skilled-waterfall-10`
- Run id: `27i1r9qi`
- State: crashed at `_step=2550` (~1h8m); SSH drop noted ~step 950 in chat; grad-skips
  from step 1500 on (grad_norm 55 → 204)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/27i1r9qi

## Parent

[investigation_003](../DESCRIPTION.md)
