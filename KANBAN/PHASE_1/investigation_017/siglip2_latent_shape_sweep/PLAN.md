# Plan — SigLIP 2 latent-shape sweep

## Scope

Run the common `3 × 3` external latent-shape grid on pinned SigLIP 2. Preserve its native detailed
lattice and normalization; do not compare its raw reconstruction magnitude to V-JEPA2 or DINOv3.

## Locked lane identity

```text
encoder=siglip2_vitb16
revision=3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
precision=bf16
frame_microbatch=8
group=inv017_siglip2_latent_shape_cov_var
```

The shared fixed recipe, grid, readout, and decision rule live in
[`../SWEEP_PLAN_latent_shape.md`](../SWEEP_PLAN_latent_shape.md).

## Execution and evidence

1. Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5, including W&B login.
2. Execute this lane's [`GUIDE.md`](GUIDE.md) directly, with no separate test suite, Stage 0, or
   resource-preflight job unless the human explicitly requests one.
3. Reuse exact prior center `ufbeokj2`, then launch the other eight grid cells sequentially.
4. Read late windows within the SigLIP lane and retain every weighted geometry term.
5. Keep the center unless another shape clears the material-effect and geometry guards.

## Risks

Run 67 did not reach an absolute healthy-geometry equilibrium despite covariance and variance, so
this lane may identify a least-bad Pareto shape rather than a passing representation. `N_c` changes
covariance sample rows and `D_c` changes covariance channels; shape and realized regularization
pressure cannot be separated in this grid. One seed also leaves marginal effects initialization-
sensitive.
