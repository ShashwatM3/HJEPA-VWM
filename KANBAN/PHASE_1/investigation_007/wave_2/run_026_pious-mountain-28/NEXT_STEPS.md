# Next Steps - run 026 `pious-mountain-28`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 027 [`quiet-firebrand-25`](../run_027_quiet-firebrand-25/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — pious-mountain-28

**TIER 0 — re-run this run.** It is one of the two bookends (`n_c=64` + `n_c=256`) that decide the
latent-capacity question; the reduced re-run wave should include it. Use the exact command in
[DESCRIPTION.md](DESCRIPTION.md); harden the launch (tmux detach + step-600 tripwire) per
[Wave 2 NEXT_STEPS](../NEXT_STEPS.md) so the synchronized death can't recur.

Read alongside [classic-yogurt-29](../run_028_classic-yogurt-29/) (n_c=128) and
[earnest-dragon-25](../run_025_earnest-dragon-25/) (n_c=256) — the latent ladder it begins. Outcome metric:
`coarse_vs_copy_ratio` → <1 (with `c_effective_rank` rising) = latent capacity is the lever;
anything else = pivot to a temporal objective ([`../END_OF_WAVE_2.md`](../../END_OF_WAVE_2.md) §2.6).
