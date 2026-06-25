# Next steps — olive-terrain-11

## Why

Killed at step 3900. The centered slot loss bites at the loss level (`L_slot` ~0.01,
slot diversity spiked to ~26) but the spike decays back to ~4.6 and the axes that matter
(`c_effective_rank`, `c_cross_video_cosine`, `c_std_mean`) do not improve — the first
Goodhart signal. The natural move was to **re-run the identical config longer** to
confirm the pattern is real rather than an artifact of the early kill.

## Spawned

**Next run:** [`copper-sky-12`](../copper-sky-12/) — same config
(`--horizon-k 12 --lambda-slot 0.05 --lambda-cov 0.0027`, centered loss), run longer.
It reproduced and sharpened this run's Goodhart (rank fell to ~4.8, cosine to ~0.84,
plus a 4×10⁴ grad spike), which is what finally rejected the slot lever.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

**Watch:** `c_effective_rank` and `coarse_vs_copy_ratio` jointly — a rising slot metric
with a flat/worsening copy ratio is Goodhart, not progress.
