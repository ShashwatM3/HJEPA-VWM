# Run 011 - `olive-terrain-11`

**Current W&B run name:** `Investigation 03 · Centered slot loss · Long horizon`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `q40nq0l3` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/q40nq0l3  
**State:** `killed`  
**Created:** 2026-06-17T13:01:03Z  
**Last history step:** 3900  
**Runtime in W&B export:** 1.97h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

First centered-slot k=12 run; tests whether longer horizon plus centered slot loss improves collapse.

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

- Previous W&B run: Run 010 [`skilled-waterfall-10`](../run_010_skilled-waterfall-10/).
- Next W&B run: Run 012 [`copper-sky-12`](../run_012_copper-sky-12/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3000836590/wandb_manifest.json` (0 bytes), `artifact/3000836615/wandb_manifest.json` (0 bytes), `artifact/3019302134/wandb_manifest.json` (0 bytes), `artifact/3020317808/wandb_manifest.json` (0 bytes), `artifact/3028581477/wandb_manifest.json` (0 bytes), `artifact/3028586654/wandb_manifest.json` (0 bytes), `config.yaml` (3731 bytes), `output.log` (26092 bytes), ... 3 more |
| Logged artifacts | `run-q40nq0l3-history:v0` (wandb-history), `run-q40nq0l3-events:v4` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — olive-terrain-11

**The first centered-slot, k=12 run.** Previously logged as config-TBD; W&B has since
resolved it.

## What this run tested

First launch after the slot loss/metric centering fix (`ffc33ed`, 06-17 12:55 UTC): now
that the slot penalty actually bites, does **centered** slot loss at a mild weight lift
slot diversity and rank without Goodhart — at the harder **k=12** horizon? This is the
run [`skilled-waterfall-10`](../run_010_skilled-waterfall-10/) was *meant* to be (centered loss,
k=12) before its bug + k=4 launch drift + crash.

## Command (reconstructed from W&B config `q40nq0l3`)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

Config confirmed: `horizon_k=12`, `lambda_slot=0.05`, `lambda_cov=0.0027`,
`lambda_var=0.10`. Created 06-17 13:01 UTC — minutes after the centering commit, and the
**first** run whose `L_slot` starts well below 1.0 (~0.01–0.04), i.e. the first run where
the slot penalty is real.

## Config delta vs skilled-waterfall-10

- Code: **centered** `slot_diversity_loss` (`ffc33ed`) — loss now matches the metric
- `horizon_k`: 4 (actual) → **12**
- Same weights otherwise (`lambda_slot=0.05`, `lambda_cov=0.0027`, `lambda_var=0.10`)

## W&B

- Run name: `olive-terrain-11`
- Run id: `q40nq0l3`
- State: killed at `_step=3900` (~1h58m)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/q40nq0l3

## Parent

[investigation_003](../DESCRIPTION.md)

## Mapping note

Earlier KANBAN marked this "config TBD / mapping uncertain." Resolved via MCP: it is a
**centered-slot k=12 slot=0.05 cov=0.0027 run** — the first of the two post-centering
slot runs (olive-terrain-11 then [`copper-sky-12`](../run_012_copper-sky-12/)). copper-sky-12
(06-19) is the re-run after this one was killed at 3900; the two together are the
combined "Run 5" slot evidence.
