# Next steps — sleek-leaf-7

## Why

Stopped ~step 3500: full SSv2 did **not** fix collapse (rank 8.7 vs ~5 on tiny — 42×
more data bought ~3.7 rank points). The per-head metric (`a96c0d6`) revealed the
*dominant* failure is **within-video slot redundancy** (`c_slot_diversity_rank` 1.62/32)
+ near-uniform attention, not the cross-video feature correlation VICReg-C targets.

## The plan-divergence decision (rejected: VICReg-C-first)

The pre-registered plan was "Run B = VICReg-C alone." The team **diverted**: VICReg-C
decorrelates the 256 feature dims (the cross-video axis, rank 8.7), but the worst axis
is the 32-slot axis (1.62), which VICReg-C does not touch — and SIGReg (the planned
escalation) is on the *same* feature axis, so it wouldn't help either. Per the user's
rule ("unless the next step combats the problem arisen, fix first"), the agent's
overnight research pointed to a **slot-diversity penalty** as the established fix for
learned-query/prototype collapse. So the next run leads with slot loss, keeping VICReg-C
on gently as the secondary term. `L_cov` at λ=0 logged 20.18 here → calibrate
**λ_cov ≈ 0.0027** (5% of `L_flow`) for the adjunct.

## Spawned

**Next run:** [`serene-cloud-8`](../serene-cloud-8/) (BRIEF Run 3) — slot-diversity
loss at aggressive weight (0.25) + low-weight VICReg-C adjunct (0.0027); easy horizon k=4.

```bash
python train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 \
  --lambda-slot 0.25 --lambda-cov 0.0027
```

**Watch:** `c_slot_diversity_rank`, `c_cross_video_cosine`, `coarse_vs_copy_ratio`
together — slot metric alone is not enough.

Do not treat rank ~9 stall as pass.
