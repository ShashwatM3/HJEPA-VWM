# Next Steps - run 023 `toasty-donkey-21`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Immediate Consequence

The next controlled axis should target latent utilization or bottleneck geometry, not simply train F_c longer.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 024 [`light-universe-24`](../run_024_light-universe-24/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — toasty-donkey-21

Read as part of the weight ladder, not in isolation. The immediate continuation is
[jolly-glade-20](../run_020_jolly-glade-20/) (λ=0.2) and then [light-universe-24](../run_024_light-universe-24/)
(λ=0.5); the three together decide whether the floor is weight-bound (it isn't —
[Wave 1 OBSERVATIONS](../OBSERVATIONS.md)).

No per-run follow-up: this config is settled (floor inert, prediction unfixed). Its only role
downstream is as the low anchor that, with its siblings, eliminated the weight axis and motivated
[Wave 2](../../wave_2/DESCRIPTION.md)'s latent ladder.
