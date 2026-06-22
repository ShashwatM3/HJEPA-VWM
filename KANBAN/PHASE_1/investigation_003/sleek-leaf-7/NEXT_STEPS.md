# Next steps — sleek-leaf-7

## Why

Stopped ~step 3500: full SSv2 did **not** fix collapse (rank 8.7 vs ~5 on tiny).
Slot redundancy (`c_slot_diversity_rank` 1.62) dominant. `L_cov` at λ=0 → calibrate
**λ_cov ≈ 0.0027** for adjunct trials.

## Spawned

**Next run:** [`serene-cloud-8`](../serene-cloud-8/) (BRIEF Run 3) — slot-diversity
loss at aggressive weight + low-weight VICReg-C adjunct; easy horizon k=4.

```bash
python train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 \
  --lambda-slot 0.25 --lambda-cov 0.0027
```

**Watch:** `c_slot_diversity_rank`, `c_cross_video_cosine`, `coarse_vs_copy_ratio`
together — slot metric alone is not enough.

Do not treat rank ~9 stall as pass.
