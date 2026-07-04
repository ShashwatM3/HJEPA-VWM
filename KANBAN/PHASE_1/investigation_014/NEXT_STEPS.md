# Next Steps

## Immediate

1. Keep `logs/drift_probe/rank_results_ssv2_validation_n64_seed42.json` as the canonical first
   encoder-rank readout for the 64-video validation probe set.
2. When interpreting future `c_effective_rank`, compare it to the encoder-side structure recorded
   here, not to the raw `D_e = 1024` ceiling alone.
3. Use this investigation when designing bottleneck losses: the goal is not to copy every weak
   V-JEPA direction, but to preserve enough rank to carry the useful factors needed by Phase 1.

## Follow-Up Measurements

- Run the same rank probe on a larger validation probe set if runtime permits, for example
  `--probe-videos 128`, to check whether the pooled `e_effective_rank` estimate is stable.
- Run the same probe on `ssv2_tiny` only as a smoke comparison; do not treat tiny-subset rank as
  canonical.
- Consider adding a later probe that ranks target-window `e_plus` features as well as anchor-window
  `e_t`, to check whether future windows have the same encoder-side rank profile.

## Modeling Questions Spawned

- Should reconstruction pressure explicitly ignore or downweight weak encoder-side tail directions?
- Should the bottleneck objective preserve the top encoder covariance subspace more directly?
- Should future diagnostics include an `e`-relative utilization ratio, with clear caveats that
  `e_effective_rank` is a baseline and not a target to blindly maximize?
