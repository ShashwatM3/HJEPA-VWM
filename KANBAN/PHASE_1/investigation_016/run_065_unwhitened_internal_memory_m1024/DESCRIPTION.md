# Run 65 — Investigation 16 · Internal memory width · EGO4D M=1024

## Status

PLANNED

## W&B

- Display name: `Investigation 16 · Internal memory width · EGO4D M=1024`
- Group: `inv016_unwhitened_internal_memory_width`
- ID/URL: pending launch

## Question

Does eliminating every dimension-reducing channel map before global latent reasoning materially
beat the practical 512-wide design while preserving stronger dependence on the supplied clip code?

This is the second arm and scientific upper bound. It uses full EGO4D, no whitening, one late
`1024 -> 256` projection, three latent blocks, absolute cosine reconstruction at weight 1, and zero
variance/covariance/SIGReg/slot weights. It starts only after the 512 sibling finishes successfully.

The paired baseline is
[`../run_064_unwhitened_internal_memory_m512/`](../run_064_unwhitened_internal_memory_m512/). The shared exact
launch is [`../GUIDE.md`](../GUIDE.md).
