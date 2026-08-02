# Run 071 — DINOv3 unwhitened M=512 with covariance plus variance

## Status

FINISHED — W&B [`fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6),
15,000/15,000 training updates.

## Question

What equilibrium does the exact raw/unwhitened DINOv3 `32×256` center reach when covariance and
the variance floor are restored, and can that completed configuration serve as the center evidence
for investigation 017?

## Exact recipe

- full EGO4D, seed 42, batch 64, 15,000 steps;
- `dinov3_vitb16@5931719e67bbdb9737e363e781fb0c67687896bc`, frozen bf16, frame microbatch 32;
- native detailed lattice `(B,2048,768)` after stripping CLS plus four register tokens;
- raw/unwhitened present-only absolute cosine reconstruction;
- `M=512`, three latent blocks, `N_c=32`, `D_c=256`, decoder `512×4`;
- `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`,
  `lambda_sigreg=lambda_slot=0`;
- clean source commit `083cf8a6e87168702efe46ac6bfe485756dcb439`.

## Research role and evidence boundary

This is the exact scientific configuration of the DINO `32×256` center registered in
investigation 017. It is also a present-only run: it does not validate DINO future encoding,
`B_EMA`, `F_c`, full-mode memory, or prediction gates. The fixed diagnostic batch is
single-source, so pair and derangement metrics remain within-source.

Full read: [`ANALYSIS.md`](ANALYSIS.md).
