# Investigation 021 — Fixed-coordinate coarse-flow learning

## Status

OPEN — one full 15,000-step run is registered but not launched. No W&B ID exists yet.

The implementation is complete in commit
`0c1d343ce19258d5d575ea17a1654a46b32abf85`. The exact run protocol lives in
[`fixed_residual_coordinates_fc_only/GUIDE.md`](fixed_residual_coordinates_fc_only/GUIDE.md).

## Parent result

This investigation is the direct causal follow-up to
[`investigation_020`](../investigation_020/). Its residual arm
[`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t)
finished all 15,000 steps with a healthy DINOv3 bottleneck but no useful predictor:

- late `c_effective_rank=376.932/512`;
- late `c_cross_video_cosine=0.179245` against encoder `0.402768`;
- late `coarse_copy_loss=1.300293`, so the code retained temporal change;
- late `coarse_vs_copy_ratio=1.726204`;
- late `coarse_vs_batch_mean_ratio=1.817973`.

The representation therefore passed while `F_c` lost both to zero residual and to the batch-mean
residual. Joint training cannot distinguish whether `F_c` failed because its input/target
coordinates moved with `B` and `B_EMA`, or because the current `F_c`/rectified-flow objective
cannot learn these dynamics even in stationary coordinates.

## Question

Can the existing six-block coarse flow learn DINOv3 bottleneck residual dynamics when the
representation is held exactly fixed for the entire temporal phase?

## Experiment

Run one full residual-prediction training job from the exact same Investigation-019 present-only
checkpoint used to initialize both Investigation-020 arms:

```text
frozen encoder E
  -> frozen pretrained B gives c_t
  -> frozen exact-copy B_EMA gives c_t^EMA and c_{t+k}^EMA
  -> F_c alone learns Delta = c_{t+k}^EMA - c_t^EMA
frozen D remains available only for diagnostic readouts
```

The atomic mode is:

```text
--temporal-target residual --optimization-scope fc_only
```

`fc_only` means:

- trainable: `F_c` only;
- frozen: `B`, `B_EMA`, and `D`;
- EMA updates: disabled;
- optimized objective: `L_flow` only;
- AGC, global clipping, AdamW groups, and LR scheduling: applied only to `F_c`;
- diagnostics: still compute representation, copy, batch-mean, and reconstruction readouts;
- provenance/checkpoints: bind and repeatedly verify exact `B`, `B_EMA`, and `D` hashes.

## Why the source is Investigation 019, not the final Investigation-020 residual checkpoint

The run warm-starts from the original present-only checkpoint
`60yaqw6d`, SHA-256
`931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`.

This recreates the same learned `B`/`D`, fresh exact-copy `B_EMA=B`, fresh `F_c`, seed, data order,
and schedule that the Investigation-020 residual arm had at step 0. The causal change is therefore
whether the representation is optimized during temporal training. Starting from the final
Investigation-020 bottleneck would instead introduce a different, jointly adapted representation
and would not cleanly answer the moving-coordinate question.

## Hypotheses

### H1 — moving coordinates were the dominant obstruction

Joint `L_flow -> B,F_c` optimization forced `F_c` to chase an evolving online condition while its
detached target was produced by a lagging EMA coordinate system. With `B_EMA=B` fixed, the copy
loss and representation diagnostics should become stationary, and `F_c` should learn a stable
conditional residual field.

Strong support requires the late-window prediction gates:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

### H2 — the present `F_c` or objective cannot learn these fixed dynamics

If the representation and copy loss remain fixed but `F_c` still loses to zero residual and the
batch mean, coordinate motion was not the decisive explanation. The next lever must then live on
the predictor/objective side rather than in more bottleneck geometry work.

### Intermediate outcome

Stable late ratios below `1.0` but above the project gates would show that fixed coordinates help
without establishing a sufficient predictor. That result would motivate a separately registered
`F_c`-capacity or temporal-objective investigation; it would not count as a Phase-1 pass.

## Locked controls

Match the Investigation-020 residual arm on:

- source checkpoint and SHA-256;
- DINOv3 encoder revision/fingerprint;
- EGO4D dataset fingerprint;
- seed `42` and deterministic data order;
- `N_c=64`, `D_c=512`, bottleneck internal width `M=512`;
- six-block, eight-head `F_c` with condition dropout `0.10`;
- residual target/noise construction, horizon `12`, frame stride `2`;
- batch `64`, 15,000 steps, 1,500-step warmup, LR and AdamW schedule;
- logging, diagnostic, and checkpoint cadence;
- fixed validation population and all readouts.

Do not add SIGReg, change encoder, change horizon, enable prediction-side reconstruction, enlarge
`F_c`, alter the source representation, or shorten the scientific run.

## Registered run

| Folder | GPU | Intended W&B identity | Comparator |
|---|---:|---|---|
| [`fixed_residual_coordinates_fc_only/`](fixed_residual_coordinates_fc_only/) | 0 | `Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512` | Investigation-020 residual `3y2hxj5t` |

## Decision rule

Use full-prediction Reading Cycle A. Late values are medians over the final six diagnostic rows,
matching Investigation 020.

| Outcome | Interpretation |
|---|---|
| Both gates pass stably | Moving joint coordinates were a primary obstruction; proceed from the fixed-coordinate predictor result. |
| Both ratios are below 1 but either formal gate fails | Coordinate stationarity helped, but current `F_c`/objective remains insufficient. |
| Either ratio remains at or above 1 while the fixed-state tripwires pass | Current `F_c`/objective cannot learn these fixed residual dynamics; stop spending on bottleneck geometry. |
| Frozen hashes change, copy/representation diagnostics drift materially, wrong mode, NaN/skip spiral, OOM, or incomplete run | Invalid; fix execution without changing the scientific recipe. |
