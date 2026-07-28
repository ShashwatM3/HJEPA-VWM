# Investigation 017 — Three-encoder raw-feature latent-shape sweep

## Status

OPEN — no queue is active. The first unique V-JEPA2 arm, `16×512` (`ihiuptdp`), is terminal in
W&B state `crashed` at step 7,100; the other V-JEPA2 shapes and all SigLIP 2/DINO shape contrasts
are unlaunched.

## Question

With the settled unwhitened late-projection bottleneck held at internal width `M=512`, how should
the external abstract code allocate capacity between:

- the number of slots `N_c`; and
- the dimension of each slot `D_c`?

The same full factorial is run independently for V-JEPA2 ViT-L/16, SigLIP 2 ViT-B/16, and DINOv3
ViT-B/16:

```text
N_c ∈ {16, 32, 64}
D_c ∈ {128, 256, 512}
```

This produces 27 design cells organized as three sweep bundles. Three centers already have exact
science-recipe evidence: V-JEPA2 `32×256` is W&B `guiduvjp`, SigLIP 2 `32×256` is `ufbeokj2`,
and DINOv3 `32×256` is `fiactcw6`. They are reused rather than rerun, leaving 24 non-center
contrasts: eight per encoder. One V-JEPA2 contrast (`16×512`) has a partial crashed history but no
valid 15,000-step result.

1. [`vjepa2_latent_shape_sweep/`](vjepa2_latent_shape_sweep/)
2. [`siglip2_latent_shape_sweep/`](siglip2_latent_shape_sweep/)
3. [`dinov3_latent_shape_sweep/`](dinov3_latent_shape_sweep/)

## Settled base recipe

All arms use full EGO4D, raw/unwhitened frozen features, present-only absolute cosine
reconstruction, `M=512`, three latent blocks, decoder `512 × 4`, `lambda_recon=1`,
`lambda_var=0.5`, `lambda_cov=0.01`, no SIGReg, no slot loss, seed 42, batch 64, and the 15,000-step
schedule. Only encoder identity and the two intended shape axes differ across the complete design.

The center `32 × 256` is the latest architecture's external code. Run 66 established the geometry
recipe on V-JEPA2; Run 67 exercised the matching SigLIP 2 recipe; local Run 071 (`fiactcw6`)
completed the matching raw DINOv3 recipe. Run 69 remains DINO's no-geometry reference. Every sweep
contrast turns variance and covariance on.

## Interpretation boundary

This is three within-encoder shape sweeps, not one raw-loss contest between encoders. Each frozen
encoder defines a different target distribution, token count, temporal semantics, covariance
spectrum, and raw cosine-loss scale. Shape effects are estimated within each lane. Cross-encoder
synthesis uses normalized rank, correct-versus-shuffled dependence, stability, and compute, while
retaining the native detailed lattice of each encoder.

The fixed covariance coefficient is intentionally part of the tested recipe. Because the
covariance statistic pools `B × N_c` slot rows over `D_c` channels, its sampling and scale change
with both axes. Results therefore identify the best shape under this objective, not a
regularizer-free causal effect of nominal scalar capacity.

See [`SWEEP_PLAN_latent_shape.md`](SWEEP_PLAN_latent_shape.md) and [`GUIDE.md`](GUIDE.md).
