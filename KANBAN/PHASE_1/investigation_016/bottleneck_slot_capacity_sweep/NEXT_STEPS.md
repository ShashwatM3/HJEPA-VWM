# Next steps — bottleneck slot-capacity sweep

1. Follow [`GUIDE.md`](GUIDE.md) without adding the no-whitening or alternate-encoder deltas.
2. Launch the three EGO4D core arms from one clean commit and record each W&B ID in `DESCRIPTION.md`.
3. Confirm every arm reaches step 500; investigation 007's prior slot ladder ended in a synchronized
   whole-pod failure and supplied no capacity evidence.
4. After all runs finish, apply Reading Cycle B separately, then compare the EGO4D
   `32 -> 64 -> 128` curve using [`PLAN.md`](PLAN.md).
5. Run a 256-slot saturation arm only if the pre-registered conditional rule calls for it.

No checkpoint from another `N_c` value may be resumed into an arm. No result from this present-only
sweep is permission to claim prediction success.
