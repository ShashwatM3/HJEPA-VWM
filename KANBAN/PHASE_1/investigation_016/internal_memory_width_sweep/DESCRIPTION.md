# Investigation 016 — unwhitened internal-memory width sweep

## Status

PLANNED

## Runs

| Arm | W&B display name | W&B ID | State |
|---|---|---|---|
| `M=512` | `Investigation 16 · Internal memory width · EGO4D M=512` | pending | not launched |
| `M=1024` | `Investigation 16 · Internal memory width · EGO4D M=1024` | pending | not launched |

Both arms use W&B group `inv016_unwhitened_internal_memory_width` and execute sequentially on one
GPU. They start from fresh initialization and never resume a whitened or differently shaped
checkpoint.

## Question

After restoring raw V-JEPA features and delaying the `M -> D_c=256` channel projection until after
all three input-dependent latent blocks, does retaining 1,024 internal channels preserve enough
additional clip information to beat the practical 512-wide design?

This is a two-point sweep of the **complete bottleneck internal width** `M`, not a sweep of slot
count or external code size. The frozen V-JEPA tensor remains `(B,1024,1024)`, `N_c` remains 32,
and the decoder still receives `(B,32,256)`.

## Combined intervention and actual sweep axis

Every arm includes the same three architecture/target decisions:

1. fixed ZCA whitening is disabled;
2. memory, learned queries, cross-attention, slot self-attention, and latent MLPs all operate at
   `M` rather than being collapsed to 256 before the first read;
3. one learned `M -> 256` projection occurs only after all three read/compete/refine blocks.

Only `M=512` versus `M=1024` changes between the two paid runs. Therefore this bundle can causally
compare the two widths, but it cannot separately attribute a historical improvement to removing
whitening versus moving the projection. That combined choice is deliberate and follows the human's
decision to implement all three changes before paying for the width sweep.

## Whitening is genuinely absent

The RunPod may still contain old files such as
`logs/whiten/whiten_stats_ego4d_train_seed42.pt`. They are historical artifacts and are left
untouched. The launch commands omit `--whiten-features`, `--whiten-stats-path`, `--whiten-eps`, and
`--whiten-expected-clips`; configuration therefore resolves `whiten_features=false`, constructs no
`FeatureWhitener`, reads no whitening payload, and records a null whitening fingerprint. There is
no stats-generation, copying, inspection, or compatibility step in this run.

## Arm sizes

| Arm | Internal path | External code | Bottleneck parameters | Meaning |
|---|---|---|---:|---|
| `M=512` | `1024 -> 512 -> three 512-D latent blocks -> 256` | `32 x 256` | 18,324,739 | Practical partial compression before global reasoning. |
| `M=1024` | `1024 -> 1024 -> three 1024-D latent blocks -> 256` | `32 x 256` | 70,727,427 | Scientific upper bound with no dimension-reducing channel map before global reasoning. |

For context, the compatible default `M=256` bottleneck has 4,837,123 parameters. The 1,024 arm is
about 3.86 times the bottleneck size of 512, so a tiny loss improvement is not automatically worth
the cost.

## Fixed recipe

- full EGO4D, pinned V-JEPA2 ViT-L/16, bf16/SDPA, seed 42;
- present-only absolute cosine reconstruction, `lambda_recon=1.0`, 2,000-step ramp;
- 15,000 steps, physical batch 64, `N_c=32`, external `D_c=256`;
- three latent blocks and a 512-wide, four-block fixed-position decoder;
- `lambda_var=lambda_cov=lambda_sigreg=lambda_slot=0`;
- bottleneck/decoder LR `1e-4`, no prediction-side reconstruction, no residual target;
- one A100, arms executed sequentially as 512 then 1,024.

The regularizer losses may still be computed for diagnostics, but their zero coefficients mean
they contribute no gradient and no value to the optimized total loss.

## What counts as preservation

Raw and whitened cosine losses live in different target geometries, so their absolute values are
not directly comparable. Within this two-arm raw-feature sweep, 512 and 1,024 are directly
comparable. Against historical whitened runs, use trajectory shape and code-dependence context,
not a claim that a smaller raw number is intrinsically better.

A lower correct-code loss is treated as genuine information preservation only when it is accompanied
by a healthier correct-code-versus-shuffled-code separation: `L_recon_shuffled_c` should remain
meaningfully above `L_recon_present`, so `L_recon_video_gap` and the conditioned share of learned
improvement do not shrink. Under the current EGO4D fixed batch this remains an exact-chunk,
within-source readout, not a global cross-source claim.

Execution is in [`GUIDE.md`](GUIDE.md); the exact code changes are in
[`IMPLEMENTATION.md`](IMPLEMENTATION.md); preregistered interpretation is in [`PLAN.md`](PLAN.md).
