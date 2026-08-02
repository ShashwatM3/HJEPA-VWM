# Next Steps - run 027 `quiet-firebrand-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 028 [`classic-yogurt-29`](../run_028_classic-yogurt-29/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — quiet-firebrand-25

**Included in the 4-GPU re-run (GPU 3).** It's confounded (3 levers at once) and Wave 1 predicts the
levers won't compound — but on a 4-GPU pod the cut was λ=1.0 (zero expected information), not this.
Its keep-value: it's the wave's **only interaction-effect check** (does stacking decoder+weight+latent
do what no single axis did?) and it sits at n_c=64, doubly-covering the low latent bookend. A clean
read still requires comparing it against the single-axis `n_c=64` run ([pious-mountain-28](../run_026_pious-mountain-28/)).
Interpret only after the clean ladder; if it moves the floor while the single axes don't, that's the
interaction signal worth chasing.

End of the planned investigation_007 sweep. The investigation's forward motion now goes through the
Tier-0 n_c re-run and then the prediction-side pivot — see [Wave 2 NEXT_STEPS](../NEXT_STEPS.md) and
[`../END_OF_WAVE_2.md`](../../END_OF_WAVE_2.md) §2.6.
