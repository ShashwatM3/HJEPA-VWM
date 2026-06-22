# Next steps — elated-snowflake-15

## Why this run stops here

Config was **correct** (matched cerulean through step ~8000). Failure mode: **grad-skip
death spiral** from step 8500 — pre-clip spike → every subsequent step skipped (~5300
steps with no weight updates). Not collapse regression.

**Do not** resume from checkpoints saved at step 10000+ (optimizer state poisoned by
skip streak). Best healthy window: steps 4500–8000 → use **`phase1_step7500.pt`**.

## Spawned

**Next run:** [`drawn-elevator-16`](drawn-elevator-16/) — resume from pre-spike
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
