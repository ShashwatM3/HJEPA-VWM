# Run — drawn-elevator-16

## What this run tested

**Resume** from `phase1_step7500.pt` after `elated-snowflake-15`'s grad-skip death
spiral, with the **coarse-flow LR halved** (2e-4 → 1e-4) to test whether slower `F_c`
updates avoid the late-run instability. Hypothesis: the ~8500 break was optimizer
instability, not the collapse config — so a gentler flow LR on resume should survive it.

## Command (confirmed — resume, lr_coarse_flow=1e-4)

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

W&B history starts at `_step≈7550` (no rows below) → **confirmed a resume**, not a fresh
run. W&B config confirms `lr_coarse_flow=1e-4` (the only non-AGC run with the halved
flow LR), `lambda_var=0.5`, `horizon_k=12`, `grad_skip_threshold=50`, **no AGC**.

## W&B

- Run name: `drawn-elevator-16`
- Run id: `0n5mx3qf`
- State: finished; `_step` 7550→14950; runtime ~3h29m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0n5mx3qf

## Parent

[investigation_005](../DESCRIPTION.md)
