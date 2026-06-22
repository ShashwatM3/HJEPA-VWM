# Next steps — skilled-waterfall-10

## Why

`L_slot` glued near ~1.0 — **loss/metric mismatch**: loss used raw cosine, diagnostic
centered slots first. Run interrupted by SSH drop ~step 950; not a clean verdict on
k=12 + mild slot alone.

## Code change (before next run)

Center slots in `slot_diversity_loss` (`commit ffc33ed`).

## Spawned

**Next run:** [`copper-sky-12`](../copper-sky-12/) (BRIEF Run 5) — same CLI after
centering fix.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

**Watch:** rank vs `coarse_vs_copy_ratio` jointly — Goodhart if slot metric improves
but copy ratio degrades.
