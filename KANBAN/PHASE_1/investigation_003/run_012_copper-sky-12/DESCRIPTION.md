# Run 012 - `copper-sky-12`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `ejror834` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ejror834  
**State:** `killed`  
**Created:** 2026-06-19T09:39:56Z  
**Last history step:** 5650  
**Runtime in W&B export:** 2.41h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

Centered-slot follow-up; confirms slot-loss Goodhart behavior rather than true Phase 1 success.

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

No tracked config-highlight difference from the previous W&B run; read this as a repeat/continuation unless the preserved notes say otherwise.

## Chronological Linkage

- Previous W&B run: Run 011 [`olive-terrain-11`](../run_011_olive-terrain-11/).
- Next W&B run: Run 013 [`cerulean-snow-13`](../run_013_cerulean-snow-13/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3000836633/wandb_manifest.json` (0 bytes), `artifact/3000836773/wandb_manifest.json` (0 bytes), `artifact/3019302351/wandb_manifest.json` (0 bytes), `artifact/3020313701/wandb_manifest.json` (0 bytes), `artifact/3020317805/wandb_manifest.json` (0 bytes), `artifact/3028581465/wandb_manifest.json` (0 bytes), `artifact/3028586650/wandb_manifest.json` (0 bytes), `config.yaml` (3731 bytes), ... 4 more |
| Logged artifacts | `run-ejror834-history:v0` (wandb-history), `run-ejror834-events:v5` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — copper-sky-12

**BRIEF Run 5.** Centered slot loss — Goodhart confirmed, slot path rejected.

## What this run tested

After the centering fix, does mild slot loss help rank without Goodhart — run **longer**
than [`olive-terrain-11`](../run_011_olive-terrain-11/) (which showed the same config Goodharting
but was killed at 3900)? This is the **re-run** of olive-terrain-11 with an identical
config; together the two are the combined "Run 5" slot evidence.

## Command (verified vs W&B config `ejror834`)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

Config confirmed: `horizon_k=12`, `lambda_slot=0.05`, `lambda_cov=0.0027`,
`lambda_var=0.10` (default). Launched 06-19 09:40 UTC (chat ~8822–8878).

## Config delta vs olive-terrain-11

- **None** — identical config; this is the longer re-run after olive was killed at 3900.

## Config delta vs skilled-waterfall-10

- Code: **centered** `slot_diversity_loss` (commit `ffc33ed`)
- `horizon_k`: 4 (waterfall actual) → **12**
- `lambda_cov=0.0027` (active)

## W&B

- Run name: `copper-sky-12`
- Run id: `ejror834`
- State: killed at `_step=5650` (~2h24m); chat discussion references ~step 4300
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ejror834

## Parent

[investigation_003](../DESCRIPTION.md)
