# Run 64 — Investigation 16 · Internal memory width · EGO4D M=512

## Status

PLANNED

## W&B

- Display name: `Investigation 16 · Internal memory width · EGO4D M=512`
- Group: `inv016_unwhitened_internal_memory_width`
- ID/URL: pending launch

## Question

Can a practical 512-wide read/compete/refine stream preserve useful raw V-JEPA information before
the unchanged final `32 x 256` code?

This is the first arm and operating-point baseline for the paired width sweep. It uses full EGO4D,
no whitening, one late `512 -> 256` projection, three latent blocks, absolute cosine reconstruction
at weight 1, and zero variance/covariance/SIGReg/slot weights. It starts from scratch.

The paired 1,024 arm is
[`../run_065_unwhitened_internal_memory_m1024/`](../run_065_unwhitened_internal_memory_m1024/). The shared exact
launch is [`../GUIDE.md`](../GUIDE.md).
