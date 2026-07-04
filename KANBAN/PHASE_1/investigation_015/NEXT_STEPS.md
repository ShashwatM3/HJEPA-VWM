# Next Steps - investigation_015

## Immediate

1. Launch run 054 `ae_latent_stack_whiten_recon_only` per [`GUIDE.md`](GUIDE.md)
   (includes the one-time `whiten_stats.py` prerequisite on the pod).
2. Analyze with Reading Cycle B (present-recon-only) plus the per-change attribution
   matrix in [`README.md`](README.md).

## Contingent follow-ups (pick by the README's 2x2 outcome)

- Feature rank holds but slots still merge -> whitening is the working lever; consider
  keeping whitening and re-testing slot competition strength (more latent blocks) or a
  small `lambda_slot`.
- Slots differentiate but feature rank contracts -> sweep `--whiten-eps` (floor may be
  amplifying tail noise or under-equalizing) before adding geometry terms back.
- Both hold -> transfer the recipe into a full-prediction run (the standing Phase 1
  gate: `coarse_vs_copy_ratio` / `coarse_vs_batch_mean_ratio`).
- Both collapse -> rerun with inv011-style geometry terms (`lambda_var`, small
  `lambda_cov`) on top, keeping whitening + residual target; the ambiguity tiebreaker is
  one bottleneck-only control run (drop the two `--whiten-*` flags, ~6 h).
