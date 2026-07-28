# Plan — V-JEPA2 latent-shape sweep

## Scope

Run the investigation-level `3 × 3` `N_c × D_c` grid on the pinned V-JEPA2 substrate. The common
implementation already exposes both axes through `--n-c` and `--d-c`; no additional training-code
mechanism is introduced by this lane.

## Locked lane identity

```text
encoder=vjepa2_vitl16
revision=b3c1679b7c34d3255ef3547f27c7b226aefab26f
precision=bf16
frame_microbatch=8
group=inv017_vjepa2_latent_shape_cov_var
```

All other fixed settings, arm order, effect thresholds, geometry normalization, and source-batch
limitation are defined in [`../SWEEP_PLAN_latent_shape.md`](../SWEEP_PLAN_latent_shape.md).

## Execution and evidence

1. Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5, including W&B login.
2. Execute this lane's [`GUIDE.md`](GUIDE.md) directly, with no separate test suite, Stage 0, or
   resource-preflight job unless the human explicitly requests one.
3. Reuse exact prior center `guiduvjp` and launch the eight non-center paid arms sequentially. The
   `16×512` cell restarts from scratch because `ihiuptdp` is an incomplete crashed arm.
4. Retain per-arm provenance and final checkpoint hashes; append W&B IDs here or in
   `OBSERVATIONS.md` as they are assigned.
5. Apply present-only Reading Cycle B and select only a material Pareto improvement over the
   `32×256` center.

## Risks

Run 66 is partial but reached step 10,950; its resolved scientific config, dataset fingerprint, and
trainable initialization are exact matches, so it is the registered center evidence rather than a
reason to pay for a duplicate. Shape changes both covariance sampling and initialization bytes, and `D_c=512` removes the
final learned projection by using the current identity path. A near-threshold result requires a
second-seed center/winner repeat.
