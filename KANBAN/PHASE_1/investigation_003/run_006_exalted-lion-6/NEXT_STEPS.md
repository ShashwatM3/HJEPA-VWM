# Next Steps - run 006 `exalted-lion-6`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Smoke / inconclusive** - The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants.
- Parent investigation next direction: The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.
- Next chronological W&B run: Run 007 [`sleek-leaf-7`](../run_007_sleek-leaf-7/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — exalted-lion-6

## Why

500-step tiny diagnostic confirmed: collapse is persistent (rank ~8, not climbing);
**attention saturation is not the bottleneck** — slot redundancy and feature
correlation are. Init fixes alone insufficient.

## Spawned

**Next run:** [`sleek-leaf-7`](../run_007_sleek-leaf-7/) (BRIEF Run 2 / VICReg Run A) — same
question on **full SSv2** with per-head entropy fix; `lambda_cov=0` (logs `L_cov` only).

```bash
python train.py --data ssv2 --steps 15000 --log-every 50 --diag-every 500
```

**Watch at ~3500:** `c_effective_rank`, `c_slot_diversity_rank`, `L_cov` (for calibrating
later slot runs). Stop early if data-confound verdict clear.

Do not treat rank ~8 as pass — target >>15.
