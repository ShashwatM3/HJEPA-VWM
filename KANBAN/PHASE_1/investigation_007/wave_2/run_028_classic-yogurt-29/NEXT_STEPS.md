# Next Steps - run 028 `classic-yogurt-29`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 029 [`helpful-snow-25`](../run_029_helpful-snow-25/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — classic-yogurt-29

**Optional re-run (deprioritized).** As the interpolation point between the decisive bookends
[pious-mountain-28](../run_026_pious-mountain-28/) (n_c=64) and [earnest-dragon-25](../run_025_earnest-dragon-25/)
(n_c=256), it adds shape but not a decision. The Tier-0 reduced wave runs only the two bookends; add
n_c=128 back only if the bookends disagree and the *shape* of the latent response matters.

If re-run, watch `c_slot_diversity_rank` / `c_cross_video_cosine` for slot collapse (its original
design-flagged risk). Continue the ladder at [earnest-dragon-25](../run_025_earnest-dragon-25/).
