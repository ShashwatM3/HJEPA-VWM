# Run 070 — DINOv3 whitened M=512 with covariance plus variance

## Status

FINISHED — W&B [`qqozribu`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/qqozribu),
15,000/15,000 training updates. The preceding 100-step operational smoke is W&B `lwx0mu34`.

## Question

Can the pinned DINOv3 substrate sustain healthy present-code geometry and code-dependent
reconstruction when fixed feature whitening, covariance, and the variance floor are enabled?

## Exact recipe

- full EGO4D, seed 42, batch 64, 15,000 steps;
- `dinov3_vitb16@5931719e67bbdb9737e363e781fb0c67687896bc`, frozen bf16, frame microbatch 32;
- native detailed lattice `(B,2048,768)` after stripping CLS plus four register tokens;
- fixed DINO whitening payload
  `72030fb7b83a3e29ca65a5bddb0d63107e6b83eba07865364b2561d558f6891c`;
- present-only absolute cosine reconstruction;
- `M=512`, three latent blocks, `N_c=32`, `D_c=256`, decoder `512×4`;
- `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`,
  `lambda_sigreg=lambda_slot=0`;
- clean source commit `083cf8a6e87168702efe46ac6bfe485756dcb439`.

The resolved `EncoderSpec`, not legacy V-JEPA compatibility fields under `model`, is the runtime
feature-geometry authority.

## Evidence boundary

This run is present-only: `prediction_active=0`, `L_flow=0`, and `L_recon_pred=0`. It validates
DINO present reconstruction, not future encoding, `B_EMA`, `F_c`, prediction memory, or the copy
and batch-mean gates. The fixed EGO4D diagnostic batch contains one source UID, so its pair cosine
and rolled-code gap are within-source measurements.

Full read: [`ANALYSIS.md`](ANALYSIS.md).
