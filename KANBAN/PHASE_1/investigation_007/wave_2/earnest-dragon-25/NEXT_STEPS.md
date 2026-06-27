# Next steps — earnest-dragon-25

**TIER 0 — re-run this run (top priority).** It is the decisive bookend of the latent axis (with
[pious-mountain-28](../pious-mountain-28/), n_c=64) and the single most informative run in Wave 2. Use
the exact command in [DESCRIPTION.md](DESCRIPTION.md).

Practical caveats for the re-run:
- **Most VRAM-hungry run** (8× slots → longer `F_c` sequence + more decoder KV). If it OOMs on
  re-run, lower its `--batch` / `num_workers` *for this run only*.
- Read the **floor ∧ `c_effective_rank` ∧ `coarse_vs_copy_ratio`** triad together to separate genuine
  enrichment from a cosmetic KV effect (see [DESCRIPTION.md](DESCRIPTION.md)).
- Watch `c_slot_diversity_rank` / `c_cross_video_cosine` — highest slot-collapse risk of the wave.

If it (and n_c=64) leave the floor flat or only cosmetically lower with `copy_ratio` still >1 → the
binding-constraint question is closed and the investigation pivots to a temporal objective
([`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md) §2.6).
