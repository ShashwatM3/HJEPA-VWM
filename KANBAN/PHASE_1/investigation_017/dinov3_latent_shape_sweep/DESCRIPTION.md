# DINOv3 latent-shape sweep

## Status

PLANNED — exact center `fiactcw6` is complete; eight non-center shapes are unlaunched.

## Question

With covariance and variance restored, which external slot/width allocation best compresses the
pinned DINOv3 dense frame lattice while preserving geometry and correct-code dependence?

## Baseline and delta

Run 69 ([W&B `it7sq8nz`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/it7sq8nz))
is the no-geometry reference. Local Run 071
([W&B `fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6)) completed the
exact raw/unwhitened geometry-active `N_c=32`, `D_c=256`, `M=512` scientific configuration.

The center's late medians are correct/rolled reconstruction `0.13382/0.17961`, gap `0.04584`,
std `0.62217`, within-source cosine `0.68489`, effective rank `69.00`, and centered slot rank
`29.03`. It is exact tried center evidence, but it fails the recorded-batch spread/alignment
thresholds. The eight new arms change only `N_c` and `D_c` inside this lane.

## Encoder contract

`dinov3_vitb16@5931719e67bbdb9737e363e781fb0c67687896bc` strips one CLS plus four register
tokens and returns eight independent 16×16 frame grids, flattened to `(B,2048,768)`. It owns
ImageNet normalization and runs frozen bf16 with validated frame microbatch 32.

Design: [`PLAN.md`](PLAN.md). Launch: [`GUIDE.md`](GUIDE.md).
