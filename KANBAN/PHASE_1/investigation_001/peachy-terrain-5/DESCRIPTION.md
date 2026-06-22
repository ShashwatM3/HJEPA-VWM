# Run — peachy-terrain-5

## What this run tested

First end-to-end Phase 1 training launch on RunPod (`ssv2_tiny`, original
pre-Run-1 hyperparameters). Isolated whether the v0.2 stack trains at all.

## Command (approximate)

```bash
python train.py --data ssv2_tiny --steps 30000
```

Pre-retune defaults: `lr_coarse_flow=4e-4`, `lr_bottleneck=2e-4`, `warmup_steps=10000`,
`grad_clip=1.0`, no `grad_skipped` guard, `lambda_var=0.10`, `horizon_k=4`.

## Config delta

Baseline first launch — no prior run on this codebase.

## W&B

- Run name: `peachy-terrain-5`
- Run id: `1chv2608`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/1chv2608

## Parent investigation

[investigation_001](../DESCRIPTION.md)
