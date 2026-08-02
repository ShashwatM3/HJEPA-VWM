# Next Steps - run 016 `drawn-elevator-16`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_005`: 15k acceptance attempts, resume behavior, AGC and skip control.
- Parent investigation next direction: The next pivot added reconstruction anchors to test whether c_t lacked usable information rather than just variance.
- Next chronological W&B run: Run 017 [`royal-cherry-17`](../run_017_royal-cherry-17/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — drawn-elevator-16

## Status

**Closed** — failed. Superseded by [`royal-cherry-17`](../run_017_royal-cherry-17/).

## Spawned

[`royal-cherry-17`](../run_017_royal-cherry-17/) — AGC on resume (grad stability achieved; new collapse mode).

## Do not

Resume from drawn-elevator checkpoints or retry halved-LR-only without AGC.
