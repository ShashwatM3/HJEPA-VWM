# Next steps — royal-cherry-17

## Status

Run **killed** at step 11350. Analysis complete. Investigation 005 **OPEN**.

## Do not

- Resume from royal-cherry checkpoints (rank 5.8, copy 12.7 @11k).
- Judge success by `grad_skipped=0` alone.
- Assume step-8500 is the break point for AGC runs — **royal broke at 8600**.

---

## Code changes (high priority)

1. **Log pre-AGC grad norm** (or raw max ratio before clip) — post-AGC norm hides the signal.
2. **Redefine `instability_warn` / abort** to fire on any of:
   - `L_flow > 1.0` for N consecutive steps (royal: sustained from 8600)
   - `agc_Fc_max_ratio > 100` (royal: 33/58 post-8500 steps)
   - `agc_B_max_ratio > 500` (royal: 6 steps, max 3229)
3. **Optional:** LR backoff when `agc_Fc_max_ratio > 50` (multiply `lr_mult` by 0.5 for 100 steps).

---

## Next run options (spawn new folder)

### Option A — Fresh 15k + AGC + halved flow LR (recommended first)

Same as royal but `--lr-coarse-flow 1e-4`. Tests whether slower F_c updates avoid the 8600 cliff.

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 \
  --log-every 50 \
  --diag-every 500
```

### Option B — Fresh 15k + tighter AGC

```bash
python train.py ... --agc-lambda-coarse-flow 0.05
```

### Option C — True resume from pre-royal checkpoint

If `phase1_step8000.pt` or `phase1_step7500.pt` exists from **elated** (not royal), resume
with AGC + halved LR. Different entry into the 8500–8600 window than royal's fresh trajectory.

**Abort rules for all options:**

- `L_flow > 1.5` for 200 consecutive steps
- `c_effective_rank` drops > 3 points in 500 steps
- `agc_Fc_max_ratio` median > 200 over any 500-step window

---

## On success

Close investigation 005 → Phase 2.
