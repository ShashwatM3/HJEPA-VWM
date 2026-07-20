# Investigation 017 — Raw-feature latent-shape sweep

## Status

OPEN — implementation and nine-arm launch queue prepared; paid runs not launched.

## Question

With Run 66's successful raw-feature geometry recipe held fixed, how should the external abstract
code allocate capacity between the number of slots `N_c` and the width of each slot `D_c`?

The sweep is the full factorial:

```text
N_c ∈ {16, 32, 64}
D_c ∈ {128, 256, 512}
```

The center control is Run 66's `32 × 256` external code. Its internal bottleneck width remains
`M=512`; 512 is not the current external slot width.

## Parent evidence

Run 66 ([W&B `guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp))
was intentionally stopped after step 10,950 but remained healthy. At the matched step 10,500,
adding covariance plus variance to the Run 64 base changed effective rank `12.26 -> 117.13`, mean
std `0.247 -> 0.877`, and within-source cross-example cosine `0.933 -> 0.437`. This establishes the
recipe as a useful sweep base without treating the partial run as a completed endpoint.

## Fixed recipe

Full EGO4D, V-JEPA2-L, raw/unwhitened features, present-only absolute cosine reconstruction,
`M=512`, three latent blocks, decoder `512 × 4`, `lambda_recon=1`, `lambda_var=0.5`,
`lambda_cov=0.01`, no SIGReg, no slot loss, seed 42, batch 64, and the 15,000-step schedule.

See [`SWEEP_PLAN_latent_shape.md`](SWEEP_PLAN_latent_shape.md) and [`GUIDE.md`](GUIDE.md).
