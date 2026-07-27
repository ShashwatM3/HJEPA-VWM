# Investigation 018 — V-JEPA2 pure-SIGReg present geometry

## Status

OPEN — one run is registered and not yet launched.

## Question

On the current raw/unwhitened Phase-1 bottleneck, can SIGReg by itself maintain a decodable,
high-rank V-JEPA2 present representation without the variance floor or covariance penalty?

The first run keeps the latest architecture and reconstruction recipe fixed:

- full EGO4D;
- present-only absolute cosine reconstruction with `lambda_recon=1`;
- internal bottleneck width `M=512`, three latent blocks, `N_c=32`, `D_c=256`;
- decoder width 512 with four blocks;
- no feature whitening and no slot loss.

It selects frozen V-JEPA2 ViT-L/16 and replaces the latest covariance-plus-variance geometry bundle
with SIGReg alone.

## Why this is a new investigation

The latest scientific run, DINOv3 W&B [`fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6),
used the current bottleneck with raw features, `lambda_var=0.5`, `lambda_cov=0.01`, and no SIGReg.
Changing both encoder family and regularizer family is not a one-variable causal comparison.

The clean within-V-JEPA2 references are:

- [`guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp): current
  `M=512` V-JEPA2 with covariance plus variance, intentionally stopped at step 10,950;
- [`4biwq87o`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o): current
  `M=512` V-JEPA2 with no geometry regularizer, finished.

No existing W&B run combines V-JEPA2, EGO4D, `M=512`, `lambda_recon=1`, no whitening, and pure
SIGReg with both variance and covariance disabled.

## Registered run

[`run_072_vjepa2_unwhitened_m512_sigreg10_only/`](run_072_vjepa2_unwhitened_m512_sigreg10_only/)
uses `lambda_sigreg=10` with the existing 2,000-step ramp. The coefficient is grounded in the
completed Investigation-011 V-JEPA2 run
[`9ap28tbw`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/9ap28tbw), where isotropy weight
10 crossed the rank-60 gate without skipped updates. That older run retained the variance floor,
used SSv2, and used the pre-`M=512` bottleneck, so it calibrates the coefficient but is not a
scientific baseline for this run.
