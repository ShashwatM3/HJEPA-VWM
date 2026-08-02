# Plan — fixed residual coordinates

## Scientific design

Run one full-scale, one-GPU residual-prediction job. Recreate the Investigation-020 residual arm's
step-0 learned representation from the same Investigation-019 warm-start checkpoint, then prevent
all representation and decoder transitions for the full run.

This isolates the optimization-scope mechanism at the intended scale. It is not a short ablation:
the paid run retains batch 64, 15,000 updates, the full EGO4D corpus, the existing six-block
`F_c`, and all standard diagnostics/checkpoint cadence.

## Implemented code contract

Implementation commit: `0c1d343ce19258d5d575ea17a1654a46b32abf85`.

One public scientific flag owns the mode:

```text
--optimization-scope {joint,fc_only}
```

The central `OptimizationPlan` resolves trainability, optimizer groups, AGC/global clipping, and
EMA behavior once for the paid path. In `fc_only`:

1. warm start validates source schema, encoder, dataset, shape, feature mode, and B/D tensor shapes
   before mutating live modules;
2. source online `B` and matched `D` load;
3. a fresh target wrapper copies online `B` exactly, establishing `B_EMA=B`;
4. `B`, `B_EMA`, and `D` are frozen; only `F_c` remains trainable;
5. the optimizer contains only `F_c` decay/no-decay groups;
6. only `L_flow` enters the optimized total;
7. AGC and global norm clipping operate only on `F_c` parameters;
8. EMA updates are disabled;
9. `B`, `B_EMA`, and `D` hashes are bound into common provenance and checked after resource
   preflight, before periodic/final checkpoints, and on resume/checkpoint load.

The decoder remains frozen and available for diagnostic reconstruction. The configured
`lambda_var`, `lambda_cov`, and `lambda_recon` values remain in provenance for recipe parity, but
those losses do not enter the `fc_only` optimized objective.

## Causal controls

Hold constant from Investigation-020 residual `3y2hxj5t`:

- source checkpoint SHA and source W&B identity;
- encoder and data fingerprints;
- seed/data ordering and fixed validation population;
- DINOv3 preprocessing and precision;
- bottleneck and flow geometry;
- residual target/noise formula;
- horizon, stride, batch, steps, optimizer hyperparameters, LR schedule, and condition dropout;
- metric/checkpoint cadence.

Change only `train.optimization_scope`. Operational identities—W&B name/ID, output path, clean
launch commit, and measured resource values—are expected to differ.

## Required gates before paid launch

1. Clean branch contains the implementation commit and exact guide.
2. One CUDA GPU, dependencies, full EGO4D data/manifests, and source checkpoint are present.
3. Full tests and production smoke tests pass.
4. Source SHA/config/fingerprint gate passes.
5. No-step provenance preflight records `fc_only`, residual target, warm-start identity, exact
   frozen hash keys, no EMA, and `L_flow`-only optimization.
6. Exact one-step resource preflight passes on GPU 0 with finite `L_flow`, no skipped update, and
   no frozen-state mutation.
7. Log, checkpoint, provenance, and tmux names do not collide with prior work.
8. `--require-wandb` is present in the paid command.

## Runtime tripwires

The launch is valid only if:

- `optimization_scope=fc_only`, `predict_residual=true`, `prediction_active=1`;
- provenance says trainable `[F_c]`, frozen `[B,B_EMA,D]`, `ema_updates=false`, objective
  `[L_flow]`;
- step-0 `loss == L_flow`, both finite, and `grad_skipped=0`;
- B/D AGC activity is zero and only `F_c` can have gradients;
- frozen hashes remain unchanged at every checkpoint boundary;
- `coarse_copy_loss`, representation geometry, and true-code reconstruction stay stationary up to
  numerical tolerance on the fixed diagnostic population.

If a frozen hash changes, stop this run as invalid and preserve the checkpoint/log/provenance. Do
not relabel a mutable-representation run as this experiment.

## Analysis plan

Use Reading Cycle A and compare final-six medians with `3y2hxj5t`.

Primary metrics:

- `coarse_vs_copy_ratio` and `coarse_vs_batch_mean_ratio`;
- `coarse_model_loss` against stationary `coarse_copy_loss`;
- `L_flow` trajectory;
- `c_effective_rank`, `c_plus_effective_rank`, `c_std_mean`, `c_cross_video_cosine` as
  stationarity checks;
- `L_recon_chat` versus fixed `L_recon_cplus` as a diagnostic of predicted endpoint quality;
- `grad_norm`, `grad_skipped`, `grad_has_nan`, and AGC metrics for validity.

Decision table and formal gates are in the parent
[`DESCRIPTION.md`](../DESCRIPTION.md). Do not infer success from falling `L_flow` alone.
