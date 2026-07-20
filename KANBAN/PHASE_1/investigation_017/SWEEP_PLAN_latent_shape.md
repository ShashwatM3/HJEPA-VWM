# Sweep plan — external latent shape

## Design

This is a `3 × 3` factorial over the two external-code axes. Everything else is copied from Run 66.

| Queue | `N_c` | `D_c` | Scalars | Role |
|---:|---:|---:|---:|---|
| 1 | 32 | 256 | 8,192 | exact center control |
| 2 | 16 | 512 | 8,192 | same capacity, width-heavy |
| 3 | 64 | 128 | 8,192 | same capacity, slot-heavy |
| 4 | 16 | 128 | 2,048 | minimum capacity |
| 5 | 16 | 256 | 4,096 | slot-count slice |
| 6 | 32 | 128 | 4,096 | width slice |
| 7 | 32 | 512 | 16,384 | width expansion at baseline slots |
| 8 | 64 | 256 | 16,384 | slot expansion at baseline width |
| 9 | 64 | 512 | 32,768 | maximum capacity |

The diagonal at 8,192 scalars isolates allocation at fixed nominal capacity. Rows and columns then
measure slot and width effects, and the corners expose interaction.

## Locked values

```text
data=ego4d, seed=42, steps=15000, batch=64, horizon=12
encoder=vjepa2_vitl16@b3c1679b7c34d3255ef3547f27c7b226aefab26f
present_recon_only=true, whiten_features=false
bottleneck_mixer_dim=512, bottleneck_latent_blocks=3
decoder_dim=512, decoder_blocks=4
lambda_recon=1.0, recon_loss_mode=cosine, recon_warmup_steps=2000
lambda_var=0.5, lambda_cov=0.01, lambda_sigreg=0, lambda_slot=0
lr_bottleneck=lr_coarse_flow=lr_decoder=1e-4
```

Only `N_c` and `D_c` change. `D_c=512` deliberately removes the final `512 -> D_c` squeeze because
`abstract_proj` becomes an identity; that is part of the width hypothesis.

## Readout and decision

For every arm, apply present-only Reading Cycle B over the last six diagnostic points and the last
50 training rows. Require zero skipped/nonfinite updates and correct mode flags.

Primary outcomes:

1. late `L_recon` and fixed `L_recon_present`;
2. `L_recon_video_gap` and conditioned share `(shuffled - correct)/(initial - correct)`;
3. `c_std_mean`, `c_cross_video_cosine`, `c_dead_dim_frac`;
4. `c_effective_rank` normalized during analysis by `min(D_c, B_val*N_c - 1)`;
5. centered slot rank normalized by `min(N_c - 1, D_c)`;
6. runtime and peak memory from the largest-shape preflight/system metrics.

Select a Pareto improvement over `32×256`: better reconstruction or code dependence without losing
healthy geometry. A lower loss with collapsed cosine/std, or higher rank with a near-zero shuffled
gap, is not a win.

## Known interpretation limit

The implemented covariance loss is held exactly fixed at `lambda_cov=0.01`, as requested. Its raw
magnitude depends on `D_c` and its pooled sample count depends on `N_c`, so the result estimates the
best shape under the settled Run-66 objective—not a geometry-loss-free causal capacity effect.
Report realized `0.01 * L_cov` for every arm.
