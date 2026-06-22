# Run — drawn-elevator-16

## What this run tested

**Resume attempt** from `phase1_step7500.pt` after `elated-snowflake-15` grad-skip death
spiral at step 8500. Expected flags: `--horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4`.

## Command (expected)

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

## W&B

- Run name: `drawn-elevator-16`
- Run id: `0n5mx3qf`
- Runtime: ~3h 29m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0n5mx3qf

## Parent

[investigation_005](../DESCRIPTION.md)

## Note

Verify resume checkpoint step and final outcome on W&B. Command inferred from chat plan
(commit `af3f87f` CLI flags).
