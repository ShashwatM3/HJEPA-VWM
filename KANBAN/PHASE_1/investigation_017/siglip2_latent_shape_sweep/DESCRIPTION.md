# SigLIP 2 latent-shape sweep

## Status

PLANNED — eight untried paid arms not launched; exact prior center `ufbeokj2` is reused.

## Question

Under the raw-feature covariance-plus-variance recipe, which external slot/width allocation best
compresses the pinned SigLIP 2 frame lattice while retaining geometry and correct-code dependence?

## Baseline and delta

Run 67 ([W&B `ufbeokj2`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ufbeokj2))
exercised the `N_c=32`, `D_c=256`, `M=512` recipe and was intentionally stopped at step 11,000
while stable. Late medians were correct reconstruction `0.228`, shuffled reconstruction `0.283`,
gap `0.055`, mean std `0.553`, pair cosine `0.766`, and effective rank about `60.5`.

W&B confirms `ufbeokj2` exactly matches the proposed center's scientific configuration, dataset
fingerprint, and trainable initialization. It supplies the center evidence; the eight untried arms
change only `N_c`/`D_c` inside this lane.

## Encoder contract

`siglip2_vitb16@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` retains only the vision patch tower and
returns eight independent 16×16 frame grids, flattened to `(B,2048,768)`. It owns SigLIP's
0.5/0.5 normalization and runs frozen bf16 with frame microbatch 8.

Design: [`PLAN.md`](PLAN.md). Launch: [`GUIDE.md`](GUIDE.md).
