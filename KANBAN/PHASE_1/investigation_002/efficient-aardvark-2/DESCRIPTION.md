# Run — efficient-aardvark-2

## What this run tested

200-step timed baseline **before** decode-only-needed-frames fix. Established
~1.66 s/step and confirmed metrics matched prior local smoke (`logs/1.json`).

## Command

```bash
time python train.py --data ssv2_tiny --steps 200
```

## Config delta

Pre-fix dataloader (decord decoded all frames then sliced). Commit `4c1abb3` era.

## W&B

- Run name: `efficient-aardvark-2`
- Run id: `fz7ztfc8`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fz7ztfc8

## Parent

[investigation_002](../DESCRIPTION.md)
