# Next Steps — run 058

1. ~~Execute the 15,000-step EGO4D transfer and analyze it.~~ Completed as W&B `mvbx96nv`; see
   [`ANALYSIS.md`](ANALYSIS.md).
2. Run the exact EGO4D recipe with `recon_residual_target=true` and keep
   `lambda_recon=0.05`. This is the GUIDE's pre-registered response to the measured near-zero
   video gap.
3. Before interpreting a reconstruction-weight sweep, add per-loss/per-module pre-clipping
   gradient and update-norm diagnostics. The current combined `grad_norm` cannot test whether B's
   reconstruction gradient is small, conflicting with geometry, or normalized away by Adam.
4. If honest residual-target reconstruction still plateaus, sweep `lambda_recon` over
   `{0.05, 0.20, 1.00}` with every other field fixed. Rank arms by video gap and representation
   gates before raw reconstruction.
