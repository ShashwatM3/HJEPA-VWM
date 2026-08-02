# Run 008 - `serene-cloud-8`

**Current W&B run name:** `Investigation 03 · Strong slot loss · Short horizon`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_003](../)  
**W&B:** `dhp1i3fk` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/dhp1i3fk  
**State:** `killed`  
**Created:** 2026-06-16T04:01:47Z  
**Last history step:** 4300  
**Runtime in W&B export:** 1.55h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Research Role

Variant in the collapse investigation; useful mainly for instability/collapse evidence.

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

| Config key | Previous run | This run |
|---|---:|---:|
| `steps` | 15000 | 5000 |
| `lambda_cov` | 0 | 0.0027 |
| `lambda_slot` | 0 | 0.25 |

## Chronological Linkage

- Previous W&B run: Run 007 [`sleek-leaf-7`](../run_007_sleek-leaf-7/).
- Next W&B run: Run 009 [`confused-butterfly-9`](../run_009_confused-butterfly-9/).
- Parent investigation conclusion: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/2987469138/wandb_manifest.json` (0 bytes), `artifact/2987469871/wandb_manifest.json` (0 bytes), `artifact/2990481080/wandb_manifest.json` (0 bytes), `artifact/2990949399/wandb_manifest.json` (0 bytes), `artifact/3000836435/wandb_manifest.json` (0 bytes), `artifact/3019302238/wandb_manifest.json` (0 bytes), `artifact/3020313698/wandb_manifest.json` (0 bytes), `artifact/3020324762/wandb_manifest.json` (0 bytes), ... 7 more |
| Logged artifacts | `run-dhp1i3fk-history:v0` (wandb-history), `run-dhp1i3fk-events:v8` (wandb-events) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — serene-cloud-8

**BRIEF Run 3.** Aggressive slot-diversity loss at easy horizon.

## What this run tested

Hypothesis: within-video **slot-diversity loss** breaks redundant slots and lifts rank.

## Why this run, not VICReg-C (the planned next step)

The plan after [`sleek-leaf-7`](../run_007_sleek-leaf-7/) was "Run B = VICReg-C / `lambda_cov`."
The team deliberately **diverted** from that. sleek-leaf-7's per-head metrics showed the
dominant collapse was **within-video slot redundancy** (`c_slot_diversity_rank` 1.62/32,
attention near-uniform), whereas VICReg-C only decorrelates the **256 feature dims**
(the cross-video axis, `c_effective_rank` 8.7). Per the user's rule — *"unless the next
step directly combats the problem(s) arisen, make the needed changes first"* — running
VICReg-C alone would have spent hours on the secondary axis. The agent's overnight
research (VICReg; Perceiver / Slot-Attention "prototype/slot collapse"; the "Trap of
Mediocrity" uniform-attention work) found the standard remedy for learned-query collapse
is a **slot-diversity / orthogonality penalty**, which attacks the root (uniform
attention) through the only available lever because `out_mlp` is shared. So a new
`slot_diversity_loss` (`losses.py`, flag-gated `--lambda-slot`, default 0, always logged
as `L_slot`) was built and tested here, with VICReg-C kept on gently as the secondary
term. A **governance flag** was raised: this exercises the supervisor's "variance floor
only, no covariance loss initially" directive — both new knobs stay flag-gated,
default-off, reversible. Decision gate set in advance: slot-rank ↑ **and** copy-ratio
holds → win; slot-rank ↑ but copy-ratio worsens → Goodhart → escalate to task difficulty.

## Command

```bash
python train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 \
  --lambda-slot 0.25 --lambda-cov 0.0027
```

(`horizon_k=4` default; `lambda_var=0.10` default.)

## Config delta vs sleek-leaf-7

- `lambda_slot=0.25` (active)
- `lambda_cov=0.0027` (active — not logging-only)

## W&B

- Run name: `serene-cloud-8`
- Run id: `dhp1i3fk`
- Runtime: ~1h 32m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/dhp1i3fk

## Parent

[investigation_003](../DESCRIPTION.md)
