# Plan — run 072 V-JEPA2 pure SIGReg

## Outcome

Launch one full-EGO4D, 15,000-update present-reconstruction run on the current bottleneck. Preserve
the latest architecture and replace covariance plus variance with SIGReg alone while selecting the
frozen V-JEPA2 adapter.

## Implementation

No production-code change is required. The current training path already supports:

- the pinned `vjepa2_vitl16` encoder;
- present-only reconstruction;
- the `M=512`, three-block late-projection bottleneck;
- deterministic per-step SIGReg sampling and a 2,000-step SIGReg ramp;
- zero-weight variance, covariance, and slot objectives;
- raw/unwhitened features.

Edit only the single authoritative `configs/train.yaml` recipe:

```text
encoder.alias = vjepa2_vitl16
model.n_c = 32
model.d_c = 256
model.bottleneck_mixer_dim = 512
model.bottleneck_latent_blocks = 3
model.decoder_dim = 512
model.decoder_blocks = 4
train.present_recon_only = true
train.lambda_recon = 1
train.lambda_recon_pred = 0
train.lambda_var = 0
train.lambda_cov = 0
train.lambda_sigreg = 10
train.lambda_slot = 0
train.sigreg_warmup_steps = 2000
train.whiten_features = false
```

The paid command hot-overrides only `data=ego4d` and `encoder=vjepa2_vitl16`; all coupled loss and
architecture fields remain reviewable in YAML.

## Gradient contract

`L_recon` and `L_sigreg` both train the online bottleneck. `L_recon` also trains the decoder.
Present-only mode leaves `F_c` unused, `B_EMA` unupdated by a future branch, and all frozen encoder
parameters outside the optimizer. With all other geometry weights at zero, SIGReg is the only
active geometry gradient.

## Coefficient decision

There is no exact prior pure-SIGReg run on EGO4D or the `M=512` bottleneck. Investigation-011
completed V-JEPA2/SSv2/no-covariance calibrations with the variance floor retained:

| W&B ID | SIGReg weight | Final effective rank | Final pair cosine | Skipped updates |
|---|---:|---:|---:|---:|
| `fq0crddc` | 7.5 | 56.81 | 0.1044 | 0 |
| `9ap28tbw` | 10 | 78.53 | 0.0951 | 0 |
| `76d6o8d2` | 12.5 | 83.41 | 0.0991 | 0 |

Weight 10 is the smallest completed setting in that ladder that clearly crossed the rank-60 gate.
The old runs used a different dataset and bottleneck and kept `lambda_var=0.5`, so this is a
grounded starting point, not a guaranteed equivalent operating pressure.

## Verification

Local:

1. Parse and assert the resolved YAML values.
2. Run `python -m py_compile` over the Phase-1 implementation.
3. Run the full test suite.
4. Run shell/Markdown checks for the new guide and KANBAN links.

Remote:

1. Verify SSH, clean worktree, exact branch/SHA, GPU, W&B authentication, and no conflicting run.
2. Fast-forward the published commit; never hand-copy the recipe.
3. Run the full tests, Stage 0, provenance preflight, and resource preflight.
4. Launch under the deterministic tmux session in [`GUIDE.md`](GUIDE.md).
5. Verify PID, GPU utilization, log, resolved config, W&B identity, and early tripwires.

## Reading and stop criteria

Use present-only Reading Cycle B.

- Q1: no skipped/nonfinite updates; `present_recon_only=1`, `prediction_active=0`, `L_flow=0`,
  `L_recon_pred=0`.
- Q2: target `c_std_mean` roughly 0.8–1.2, dead fraction near zero, pair cosine below 0.5.
- Q3: target effective rank above 60; read centered slot rank alongside it.
- Q4: reconstruction must improve after its 2,000-step ramp.
- Q5: geometry and reconstruction must improve together.
- Q6 labels the final result; present-only evidence is never a prediction claim.

Stop the paid process only for a declared operational failure: wrong resolved recipe, wrong
encoder/revision, whitening active, nonzero prediction path, repeated skipped/nonfinite updates,
NaNs, or an unrecoverable W&B/checkpoint failure. A merely disappointing scientific metric is
recorded, not tuned away mid-run.
