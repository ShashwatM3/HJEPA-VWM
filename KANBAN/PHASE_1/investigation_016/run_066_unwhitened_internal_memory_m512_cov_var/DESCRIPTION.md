# Run 66 — Investigation 16 · Raw-feature geometry · M=512 covariance plus variance

## Status

STOPPED BY USER — intentionally ended at approximately step `10,950` before the 15,000-step
endpoint on 2026-07-19 after the encoder comparison was redirected to the clean SigLIP2 control.
The run was healthy at stop time; this is not a crash or failed gate. Checkpoints and W&B history
remain preserved on the RunPod volume.

## W&B

- Display name: `Investigation 16 · Raw-feature geometry · M=512 covariance plus variance`
- Group: `inv016_unwhitened_m512_geometry`
- ID / URL: [`guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp)

## Question

Can the selected 512-wide late-projection bottleneck retain Run 64's strong raw-feature
reconstruction while explicit variance and covariance pressure prevent its low-rank,
strongly aligned code geometry?

This is the direct geometry-bundle follow-up to
[`../run_064_unwhitened_internal_memory_m512/`](../run_064_unwhitened_internal_memory_m512/).
It starts from scratch and holds full EGO4D, seed 42, data order, V-JEPA2-L, M=512, `N_c=32`,
external `D_c=256`, three latent blocks, the late `512 -> 256` projection, the 512-by-4 decoder,
present-only absolute raw-feature cosine reconstruction, `lambda_recon=1`, and the full 15,000-step
schedule fixed.

The only scientific change is the settled two-term geometry bundle:

```text
lambda_var: 0.0 -> 0.5
lambda_cov: 0.0 -> 0.01
```

Whitening, residual reconstruction, SIGReg, slot loss, future prediction, and checkpoint resume
remain off. The existing whitening-stat files on the volume are not read.

## Interpretation boundary

The current fixed EGO4D diagnostic batch contains 16 adjacent chunks from one source UID.
`c_cross_video_cosine` and the rolled-code gap therefore measure within-source cross-example
behavior here; they do not prove global cross-source separation or collapse.
