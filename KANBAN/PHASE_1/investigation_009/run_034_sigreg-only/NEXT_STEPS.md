# Next Steps - run 034 `sigreg-only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_009`: residual prediction and zero-change baseline diagnosis.
- Parent investigation next direction: The next branch cleaned optimizer and regularization settings around residual prediction to see whether the result survived a full clean run.
- Next chronological W&B run: Run 035 [`sigreg-recon-residual`](../run_035_sigreg-recon-residual/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — fine-meadow-34 (Run 1)

The substrate arm is complete; no standalone follow-up run. Its evidence feeds two decisions at the
investigation level ([../NEXT_STEPS.md](../NEXT_STEPS.md)):

1. **SIGReg-only is not a clean prediction substrate.** Rank 50 but ρ → ~0.92, slot-rank 5.4/32, and
   cross-video cosine rising to 0.39. Any going-forward base built on strong SIGReg alone inherits a
   static, slot-redundant `c`. The "adopt SIGReg-only as the base" branch in
   [../NEXT_STEPS.md](../NEXT_STEPS.md) Tier 2 reads as **not selected** on this evidence.
2. **It anchors the ρ analysis** as the static pole against [graceful-river-35](../run_035_sigreg-recon-residual/).
   Together they motivate governing the temporal autocorrelation ρ(c_t, c_{t+k}) into the predictive
   band — the leading thread into a probable **investigation_010** (anti-collapse on Δ̂, recon/`k`
   tuning to hold ρ ≈ 0.77).

No further runs branch directly from this one.
