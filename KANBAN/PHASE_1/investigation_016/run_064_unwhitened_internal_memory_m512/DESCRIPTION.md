# Run 64 — Investigation 16 · Internal memory width · EGO4D M=512

## Status

COMPLETE — 15,000/15,000 steps; W&B finished; final checkpoint verified.

## W&B

- Display name: `Investigation 16 · Internal memory width · EGO4D M=512`
- Group: `inv016_unwhitened_internal_memory_width`
- ID: `4biwq87o`
- URL: <https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o>

## Question

Can a practical 512-wide read/compete/refine stream preserve useful raw V-JEPA information before
the unchanged final `32 x 256` code?

This is the first arm and operating-point baseline for the paired width sweep. It uses full EGO4D,
no whitening, one late `512 -> 256` projection, three latent blocks, absolute cosine reconstruction
at weight 1, and zero variance/covariance/SIGReg/slot weights. It starts from scratch.

The paired 1,024 arm is
[`../run_065_unwhitened_internal_memory_m1024/`](../run_065_unwhitened_internal_memory_m1024/). The shared exact
launch is [`../GUIDE.md`](../GUIDE.md).

Run 64 trained stably and learned a materially code-dependent raw-feature reconstruction, but the
recorded within-source latent geometry contracted strongly when every geometry weight was set to
zero. The completed paired arm improved late `L_recon` by only `0.000906` and shuffled gap by only
`0.002029`, so the preregistered cost rule selects M=512. Historical whitened loss remains an
invalid numerical control for this raw-target bundle. See
[`OBSERVATIONS.md`](OBSERVATIONS.md) for the complete reading-cycle evidence and scope limits.
