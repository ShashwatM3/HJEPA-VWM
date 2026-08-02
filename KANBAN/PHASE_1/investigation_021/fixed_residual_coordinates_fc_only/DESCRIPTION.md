# Fixed residual coordinates — coarse-flow-only training

## Status

RUNNING — launched from clean commit `54a207cf9193404401c2c36c3eaf8be09039167c`.

W&B: [`r0s6ouwd`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/r0s6ouwd).

## Hypothesis

The Investigation-020 residual arm failed because `F_c` learned against coordinates that were
still changing under joint optimization. Replaying the same residual task from the same pretrained
step-0 bottleneck while freezing `B`, exact-copy `B_EMA`, and `D` should turn it into a stationary
conditional-flow problem. If coordinate motion was the dominant obstruction, `F_c` should beat
both zero residual and the batch-mean residual in the late window.

The falsifier is direct: if frozen-state and stationarity checks pass but either late ratio remains
at or above `1.0`, the current `F_c`/flow objective cannot learn these dynamics in fixed
coordinates.

## Source identity

This is the exact Investigation-019 source used by Investigation 020, not an Investigation-020
terminal checkpoint.

| Field | Locked value |
|---|---|
| source W&B ID | `60yaqw6d` |
| source checkpoint | `/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt` |
| source checkpoint SHA-256 | `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1` |
| feature fingerprint | `963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415` |
| dataset fingerprint | `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c` |
| encoder | pinned `dinov3_vitb16` |
| shape | `N_c=64`, `D_c=512`, `M=512` |

Warm start loads source online `B` and matched `D`, initializes fresh `B_EMA=B`, and keeps `F_c`,
optimizer, schedule, step, sampler, RNG, W&B ID, checkpoints, and provenance fresh. The
`fc_only` scope then freezes `B`, `B_EMA`, and `D` before the first optimizer step.

## Config delta from Investigation-020 residual run `3y2hxj5t`

```text
train.optimization_scope: joint -> fc_only
```

The temporal target remains residual. No architecture, data, seed, source, schedule, horizon,
regularizer coefficient, or diagnostic changes are permitted.

The effective optimization contract changes from joint `B/F_c/D` plus EMA to:

```text
trainable_modules: [F_c]
frozen_modules: [B, B_EMA, D]
ema_updates: false
optimized_objective_terms: [L_flow]
```

## Intended W&B identity

```text
entity: smahalanobis-uc-davis
project: hjepa-vwm
group: inv021_fixed_residual_coordinates
name: Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512
```

## Output identity

```text
tmux: inv021_fc_only
log: /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log
checkpoint root: /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only
provenance: /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/run_provenance.json
```

## Comparator and evaluation

Primary comparator: Investigation-020 residual run
[`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t).

Use full-prediction Reading Cycle A and final-six diagnostic medians. Formal success requires:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

The full 15,000-step result is required. The resource preflight is an execution gate, not a small
scientific ablation.
