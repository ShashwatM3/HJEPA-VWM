# Run 069 — DINOv3 unwhitened M=512 without geometry regularizers

## Status

FINISHED — W&B [`it7sq8nz`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/it7sq8nz),
15,000/15,000 training steps.

## Question

Can the pinned DINOv3 ViT-B/16 substrate run the settled raw-feature, late-projection
present-only pipeline end to end, and what representation/reconstruction equilibrium does it
reach without variance or covariance pressure?

## Exact recipe

- full EGO4D, seed 42, batch 64, 15,000 steps;
- `dinov3_vitb16@5931719e67bbdb9737e363e781fb0c67687896bc`, bf16, frame microbatch 32;
- native time-major detailed lattice `(B,2048,768)` after stripping CLS plus four registers;
- present-only absolute cosine reconstruction, raw/unwhitened features;
- `M=512`, three latent blocks, `N_c=32`, `D_c=256`, decoder `512 × 4`;
- `lambda_recon=1`, `lambda_var=lambda_cov=lambda_sigreg=lambda_slot=0`;
- bottleneck, coarse-flow, and decoder peak learning rates all `1e-4`.

The run used clean source commit `083cf8a6e87168702efe46ac6bfe485756dcb439`. The resolved
`EncoderSpec`, not the legacy V-JEPA compatibility fields still serialized under `model`, is the
runtime feature-geometry authority.

## Research role

This is a substrate/path validation and no-geometry baseline for DINOv3. It is not a direct
quality comparison with V-JEPA2 or SigLIP 2: each encoder defines a different raw target space,
so their cosine reconstruction levels and raw rank spectra are not commensurate.

Full read: [`ANALYSIS.md`](ANALYSIS.md).
