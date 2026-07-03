# Next Steps - run 025 `earnest-dragon-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis.
- Parent investigation next direction: That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.
- Next chronological W&B run: Run 026 [`pious-mountain-28`](../run_026_pious-mountain-28/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — earnest-dragon-25

**TIER 0 — re-run this run (top priority).** It is the decisive bookend of the latent axis (with
[pious-mountain-28](../run_026_pious-mountain-28/), n_c=64) and the single most informative run in Wave 2. Use
the exact command in [DESCRIPTION.md](DESCRIPTION.md).

Practical caveats for the re-run:
- **Most VRAM-hungry run** (8× slots → longer `F_c` sequence + more decoder KV). If it OOMs on
  re-run, lower its `--batch` / `num_workers` *for this run only*.
- Read the **floor ∧ `c_effective_rank` ∧ `coarse_vs_copy_ratio`** triad together to separate genuine
  enrichment from a cosmetic KV effect (see [DESCRIPTION.md](DESCRIPTION.md)).
- Watch `c_slot_diversity_rank` / `c_cross_video_cosine` — highest slot-collapse risk of the wave.

If it (and n_c=64) leave the floor flat or only cosmetically lower with `copy_ratio` still >1 → the
binding-constraint question is closed and the investigation pivots to a temporal objective
([`../END_OF_WAVE_2.md`](../../END_OF_WAVE_2.md) §2.6).
