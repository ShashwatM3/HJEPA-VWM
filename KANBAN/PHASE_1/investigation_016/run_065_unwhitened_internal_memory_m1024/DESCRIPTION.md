# Run 65 — Investigation 16 · Internal memory width · EGO4D M=1024

## Status

COMPLETE — 15,000/15,000 steps; W&B finished; final checkpoint verified.

## W&B

- Display name: `Investigation 16 · Internal memory width · EGO4D M=1024`
- Group: `inv016_unwhitened_internal_memory_width`
- ID: `8gr3je5b`
- URL: <https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8gr3je5b>

## Question

Does eliminating every dimension-reducing channel map before global latent reasoning materially
beat the practical 512-wide design while preserving stronger dependence on the supplied clip code?

This is the second arm and scientific upper bound. It uses full EGO4D, no whitening, one late
`1024 -> 256` projection, three latent blocks, absolute cosine reconstruction at weight 1, and zero
variance/covariance/SIGReg/slot weights. It starts only after the 512 sibling finishes successfully.

The paired baseline is
[`../run_064_unwhitened_internal_memory_m512/`](../run_064_unwhitened_internal_memory_m512/). The shared exact
launch is [`../GUIDE.md`](../GUIDE.md).

Run 65 trained stably and retained materially more external-code rank than the 512-wide arm, but
the added rank produced only a `0.000906` improvement in late active `L_recon` (`0.00148` on the
fixed correct-code diagnostic) and a `0.00203` larger correct-versus-shuffled gap. Both
preregistered material-effect thresholds are missed, so the paired result selects 512 as the
practical internal width. See
[`OBSERVATIONS.md`](OBSERVATIONS.md) for the complete Reading Cycle B evidence and scope limits.
