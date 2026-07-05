# Next Steps - run 041 `inv011_fixed_position_present_recon`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank decodable** - The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots.

## Immediate Consequence

Treat reconstruction as real but under-constrained. The next run should target rank/slot diversity without sacrificing the reconstruction loss.

## Linkage To The Research Chain

- This run belongs to `investigation_011`: cosine reconstruction, fixed-position decoder, present-only geometry sweeps.
- Parent investigation next direction: The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.
- Next chronological W&B run: Run 042 [`po_geom_sig7p5_cov0`](../present-only-geometry-sweep/wave_1/run_042_po_geom_sig7p5_cov0/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** run 041 is the present-only anchor (control) for the SIGReg x covariance geometry
sweep: fixed-position decoder + cosine + var 0.5 + SIGReg 5, no future branch, reaching rank ~49.6,
cross-video cosine 0.09, L_recon_present 0.345 — a decodable, spread, video-specific present code
(verdict Low-rank decodable; rank still near the slot-bounded band). It directly spawned the 10-run
present-only geometry sweep (runs 042-051) that added covariance pressure and pushed rank past 100
(up to ~150). Full analysis:
[`ANALYSIS_inv011_fixed_position_present_recon.md`](ANALYSIS_inv011_fixed_position_present_recon.md).
