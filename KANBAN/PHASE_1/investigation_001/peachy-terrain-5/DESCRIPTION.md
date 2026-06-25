# Run — peachy-terrain-5

## What this run tested

First end-to-end Phase 1 training launch on RunPod (`ssv2_tiny`, original
pre-Run-1 hyperparameters). Isolated whether the v0.2 stack trains at all.

## Command

```bash
python train.py --data ssv2_tiny --steps 30000
```

Command confirmed from the pod launch (`cursor_messages` ~1684/2133) and W&B config
(`max_steps=30000`). All hyperparameters came from `config.py` defaults at the
pre-Run-1 (commit `5cb8330`/`e1e1956`) era — there were no override flags yet beyond
`--data`/`--steps`.

Pre-retune defaults (verified vs W&B config `1chv2608`): `lr_coarse_flow=4e-4`,
`lr_bottleneck=2e-4`, `warmup_steps=10000`, `grad_clip=1.0`, **no `grad_skip_threshold`**
(the skip guard did not exist yet), `lambda_var=0.10`, `horizon_k=4`, `precision=bf16`,
`stage1_steps=30000`, `checkpoint_every=5000`. This is the **only** run in the project
with the 30k/10k-warmup schedule, `grad_clip=1.0`, and the un-halved LRs — every
later run uses the post-crash retune (`611f2cd`).

## Config delta

Baseline first launch — no prior run on this codebase.

## W&B

- Run name: `peachy-terrain-5`
- Run id: `1chv2608`
- State: **failed** at `_step≈10950` (NaN); runtime ~3h53m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/1chv2608

## Parent investigation

[investigation_001](../DESCRIPTION.md)
