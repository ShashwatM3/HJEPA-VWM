# Observations — peachy-terrain-5

## Key numbers (diagnostic batches)

| Step | `c_effective_rank` | `coarse_vs_copy_ratio` | `grad_norm` | Notes |
|---|---|---|---|---|
| 500 | 5.29 | 3.27 | 0.51 | First diag |
| 1000 | 10.48 | 12.93 | 2.44 | Rank spike (noise) |
| 8500 | 4.97 | **2.29** | 2.64 | Best copy ratio of run |
| 9500 | 5.30 | 7.90 | 0.97 | Ratio regressed |
| 10500 | 4.90 | 7.94 | 3.72 | Last clean diag |
| 10550 | — | — | **864** | Explosion begins |
| 10750 | — | — | NaN | Weights corrupted |

Also: `c_std_mean` healthy (~0.74–0.86), `c_dead_dim_frac=0` throughout healthy phase.

## What broke

Gradient explosion at peak LR → NaN weights → crash on NaN covariance in diagnostics.

## What worked

Pipeline, W&B, EMA, frozen encoder, variance floor all functioned until the explosion.
Copy ratio showed a real downward trend early (12.93 → 2.29) before late regression.

## Interpretation

Training dynamics were unstable at peak LR; representational collapse (`rank ~5`)
was a **separate** problem visible throughout. Stability fixes do not fix collapse.
