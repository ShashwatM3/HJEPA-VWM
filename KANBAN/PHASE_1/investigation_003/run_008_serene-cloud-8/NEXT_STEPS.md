# Next Steps - run 008 `serene-cloud-8`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants.
- Parent investigation next direction: The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.
- Next chronological W&B run: Run 009 [`confused-butterfly-9`](../run_009_confused-butterfly-9/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — serene-cloud-8

## Why

Early Goodhart: with aggressive `lambda_slot=0.25` the latent drifted toward
video-independence (`c_cross_video_cosine` ~0.70–0.78) and the model stayed 5–15× worse
than copy, even though `c_effective_rank` rose to ~14 (so the rejection is about
semantics + copy ratio, **not** rank — see OBSERVATIONS for the correction). The easy
k=4 horizon also leaves a strong copy baseline (context/target overlap ~75%).

## What was debated next (rejected alternatives)

The tech lead proposed a three-part fix: (1) `horizon_k=16` to remove context/target
overlap, (2) drop slot weight 0.25 → 0.025, (3) keep VICReg-C gentle/off. The user
pushed back on all three: was k=16 too aggressive for the diagnostics to measure
robustly across 16 frames? Is the slot metric calculation even trustworthy (don't treat
metrics as ground truth)? Is 0.25→0.025 too big a jump? After re-checking the code, the
team settled on a **more conservative** combination: **`horizon_k=12`** (harder than 4,
but the metrics still resolve it), **`lambda_slot=0.05`** (between 0.25 and 0.025), and
keep `lambda_cov=0.0027`. `k=16` was rejected as over-aggressive; bare 0.025 was rejected
as too weak given slot collapse is real.

## Spawned

**Next run:** [`skilled-waterfall-10`](../run_010_skilled-waterfall-10/) (BRIEF Run 4) — harder
horizon (k=12 intended) + gentler slot loss 0.05, keep `lambda_cov=0.0027`.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

**Watch:** whether `L_slot` actually moves — loss/metric alignment was already suspect
here (raw cosine), and waterfall-10 is where that suspicion gets confirmed as a bug.
