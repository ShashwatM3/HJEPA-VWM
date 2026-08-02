# Next Steps - run 010 `skilled-waterfall-10`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants.
- Parent investigation next direction: The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.
- Next chronological W&B run: Run 011 [`olive-terrain-11`](../run_011_olive-terrain-11/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — skilled-waterfall-10

## Why

`L_slot` glued near ~1.0 — **loss/metric mismatch**: loss used raw cosine, diagnostic
centered slots first. Run interrupted by SSH drop ~step 950; not a clean verdict on
k=12 + mild slot alone.

## Code change (before next run)

Center slots in `slot_diversity_loss` (`commit ffc33ed`).

## Spawned

**Next run:** [`copper-sky-12`](../run_012_copper-sky-12/) (BRIEF Run 5) — same CLI after
centering fix.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

**Watch:** rank vs `coarse_vs_copy_ratio` jointly — Goodhart if slot metric improves
but copy ratio degrades.
