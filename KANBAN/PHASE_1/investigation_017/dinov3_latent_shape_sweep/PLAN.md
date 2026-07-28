# Plan — DINOv3 latent-shape sweep

## Scope

Run the eight untried cells of the common `3 × 3` external latent-shape grid on the validated
pinned DINOv3 adapter. W&B `fiactcw6` already supplies the exact geometry-active `32×256` center.

## Locked lane identity

```text
encoder=dinov3_vitb16
revision=5931719e67bbdb9737e363e781fb0c67687896bc
precision=bf16
frame_microbatch=32
group=inv017_dinov3_latent_shape_cov_var
```

The shared fixed recipe, grid, readout, and decision rule live in
[`../SWEEP_PLAN_latent_shape.md`](../SWEEP_PLAN_latent_shape.md).

## Execution and evidence

1. Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5, including W&B login.
2. Execute this lane's [`GUIDE.md`](GUIDE.md) directly, with no separate test suite, Stage 0, or
   resource-preflight job unless the human explicitly requests one.
3. Reuse center `fiactcw6`; do not launch another `32×256` seed-42 duplicate.
4. Launch the eight non-center arms sequentially and retain provenance/checkpoint hashes.
5. Read all results within DINOv3; use Run 69 only to describe the effect of restoring geometry at
   the center, not to rank shapes.

## Risks

The gated checkpoint must remain available on the host. DINO is framewise, so present-only success
cannot establish temporal forecasting. Covariance/variance may impose a substantial reconstruction
tax relative to Run 69; that tax is acceptable only if code geometry becomes materially healthier
and correct-versus-shuffled separation remains positive. Shape-dependent initialization and
regularizer scale require a repeat for marginal conclusions.
