# Investigation 022 — coarse-flow architecture and metric audit

## Status

**OPEN — reasoning audit; no training run is registered.**

## Parent evidence

- Investigation 020: joint residual prediction preserved a healthy representation but did not beat
  copy or batch mean.
- Investigation 021: freezing `B`, `B_EMA`, and `D` improved the predictor by about 12.5%, but the
  late copy and batch-mean ratios remained `1.488469` and `1.575406`.
- W&B run `r0s6ouwd`: fixed coordinates, healthy representation, honest fixed decoder, stable
  optimization, and a long predictor plateau.

## Question

Before paying for another experiment, determine which claims the current evidence actually
supports:

1. Would increasing reconstruction weight constrain the predicted future to a useful latent
   manifold?
2. Is `coarse_vs_copy_ratio` calculated incorrectly, or is its interpretation too strong?
3. Does the current `F_c` repeat the old bottleneck's early-projection mistake?
4. Is there a different architectural reason why `F_c` can learn its loss yet stop near the same
   baseline ratio?

## Scope

This investigation changes no code and launches no run. It records the first-principles answer and
the smallest decisive diagnostic sequence. Its purpose is to prevent a new paid run from mixing up
three different possibilities: ignored conditioning, a restrictive output parameterization, and a
misinterpreted evaluation baseline.
