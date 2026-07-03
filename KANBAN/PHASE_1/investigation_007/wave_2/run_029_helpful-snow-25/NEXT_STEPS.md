# Next Steps - run 029 `helpful-snow-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 030 [`lambda_sigreg_3.0`](../../../investigation_008/run_030_lambda_sigreg_3.0/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — helpful-snow-25

**DROPPED from the 4-GPU re-run (2026-06-27).** With Wave 1's clean −0.005/doubling weight slope,
the λ=1.0 outcome (~0.581, no break) is near-certain, and the live W&B pull confirmed `L_flow`
*didn't even degrade* at λ=0.5 (≈0.42, same as λ=0.1) — so the one thing this run might have shown
won't appear either. It is the single most predictable run in the wave (zero expected information),
so with only 4 GPUs it's the cut: the slots go to the full `n_c` ladder (64/128/256) + the combined
run instead. Rationale: [Wave 2 NEXT_STEPS](../NEXT_STEPS.md), [`../../GUIDE.md`](../../GUIDE.md) §3c.

Resurrect only if a later result makes the weight extreme worth closing on the record, or to observe
the recon-vs-prediction trade-off directly — neither is currently warranted.

Its lasting value is forensic, not experimental: it is the architectural control proving the Wave-2
death was an external pod event, not a code/`n_c` bug (see [OBSERVATIONS.md](OBSERVATIONS.md)).
