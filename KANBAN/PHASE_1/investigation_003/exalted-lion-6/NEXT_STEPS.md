# Next steps — exalted-lion-6

## Why

500-step tiny diagnostic confirmed: collapse is persistent (rank ~8, not climbing);
**attention saturation is not the bottleneck** — slot redundancy and feature
correlation are. Init fixes alone insufficient.

## Spawned

**Next run:** [`sleek-leaf-7`](../sleek-leaf-7/) (BRIEF Run 2 / VICReg Run A) — same
question on **full SSv2** with per-head entropy fix; `lambda_cov=0` (logs `L_cov` only).

```bash
python train.py --data ssv2 --steps 15000 --log-every 50 --diag-every 500
```

**Watch at ~3500:** `c_effective_rank`, `c_slot_diversity_rank`, `L_cov` (for calibrating
later slot runs). Stop early if data-confound verdict clear.

Do not treat rank ~8 as pass — target >>15.
