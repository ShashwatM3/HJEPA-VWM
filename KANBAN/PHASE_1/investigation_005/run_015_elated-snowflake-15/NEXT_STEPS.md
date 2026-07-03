# Next Steps - run 015 `elated-snowflake-15`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Invalid** - Training-health signals failed; later metrics should not be treated as reliable evidence.

## Immediate Consequence

Use the failure mode identified above to select one controlled next axis; do not combine unrelated changes before the baseline gates are understood.

## Linkage To The Research Chain

- This run belongs to `investigation_005`: 15k acceptance attempts, resume behavior, AGC and skip control.
- Parent investigation next direction: The next pivot added reconstruction anchors to test whether c_t lacked usable information rather than just variance.
- Next chronological W&B run: Run 016 [`drawn-elevator-16`](../run_016_drawn-elevator-16/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

# Next steps — elated-snowflake-15

## Why this run stops here

Config was **correct** (matched cerulean through step ~8000). Failure mode: **grad-skip
death spiral** from step 8500 — pre-clip spike → every subsequent step skipped (~5300
steps with no weight updates). Not collapse regression.

**Do not** resume from checkpoints saved at step 10000+ (optimizer state poisoned by
skip streak). Best healthy window: steps 4500–8000 → use **`phase1_step7500.pt`**.

## Spawned

**Next run:** [`drawn-elevator-16`](../run_016_drawn-elevator-16/) — resume from pre-spike
checkpoint with **halved coarse-flow LR** (2e-4 → 1e-4). Same collapse config.

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

**Why halve LR:** spike at 8500 was optimizer instability late in run, not wrong
`lambda_var` or horizon. Lower flow LR reduces spike risk on resume.

**Watch:** `grad_skipped`, `coarse_vs_copy_ratio`, `c_effective_rank`.

**Abort** if `grad_skipped=1` sustained (>10% over 500 steps).

**Operational rule added:** sustained `grad_skipped` → stop run.

This run does **not** close investigation 005 — it hands off to `drawn-elevator-16`.
