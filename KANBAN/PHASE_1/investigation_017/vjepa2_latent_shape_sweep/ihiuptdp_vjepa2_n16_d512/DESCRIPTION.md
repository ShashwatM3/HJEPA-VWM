# Investigation 17 · V-JEPA2 latent shape · N=16, D=512

## Status

CRASHED — W&B [`ihiuptdp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ihiuptdp),
last logged training step 7,100. No final 15,000-step checkpoint is recorded in W&B.

## Question and recipe

This is the width-heavy equal-capacity contrast to center `guiduvjp`: V-JEPA2 raw/unwhitened
features, `M=512`, `N_c=16`, `D_c=512`, covariance plus variance, present-only absolute cosine
reconstruction, seed 42, and the common 15,000-step schedule. Its clean source commit is
`771cbba077d9f846bdf7a7dd48e12dbf29d54b49`.

## Evidence boundary

The partial history can diagnose execution and trajectory, but it is not a completed sweep cell.
W&B reports the terminal state and zero logged training tripwires; it does not identify the
external crash cause.
