# Run — youthful-pond-1

## What this run tested

First end-to-end training smoke on RunPod after initial setup: does the v0.2 stack
launch, log to W&B, and produce sane step-0 metrics on `ssv2_tiny`?

## Command

```bash
python train.py --data ssv2_tiny --steps 100
```

## Config delta

Baseline first launch — no prior run on this codebase.

## W&B

- Run name: `youthful-pond-1`
- Run id: `x4pwz33d`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/x4pwz33d

## Parent

[investigation_002](../DESCRIPTION.md)
