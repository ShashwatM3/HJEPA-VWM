# Next Steps - run 011 `olive-terrain-11`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants.
- Parent investigation next direction: The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.
- Next chronological W&B run: Run 012 [`copper-sky-12`](../run_012_copper-sky-12/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — olive-terrain-11

## Why

Killed at step 3900. The centered slot loss bites at the loss level (`L_slot` ~0.01,
slot diversity spiked to ~26) but the spike decays back to ~4.6 and the axes that matter
(`c_effective_rank`, `c_cross_video_cosine`, `c_std_mean`) do not improve — the first
Goodhart signal. The natural move was to **re-run the identical config longer** to
confirm the pattern is real rather than an artifact of the early kill.

## Spawned

**Next run:** [`copper-sky-12`](../run_012_copper-sky-12/) — same config
(`--horizon-k 12 --lambda-slot 0.05 --lambda-cov 0.0027`, centered loss), run longer.
It reproduced and sharpened this run's Goodhart (rank fell to ~4.8, cosine to ~0.84,
plus a 4×10⁴ grad spike), which is what finally rejected the slot lever.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

**Watch:** `c_effective_rank` and `coarse_vs_copy_ratio` jointly — a rising slot metric
with a flat/worsening copy ratio is Goodhart, not progress.
