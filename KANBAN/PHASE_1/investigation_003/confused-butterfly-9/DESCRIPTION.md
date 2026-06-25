# Run — confused-butterfly-9

## What this run tested

Nothing useful — **immediate failure** (0s compute, no metrics logged). A mis-launch of
the slot-loss experiment: W&B config matches the planned **k=4, slot=0.25** slot run
(`horizon_k=4`, `lambda_slot=0.25`, `lambda_cov=0.0027`, `max_steps=5000`,
`diag_every=250`). It was created 06-16 18:15 UTC, **one second before**
[`skilled-waterfall-10`](../skilled-waterfall-10/) (18:16) — i.e. a fat-finger / instant
re-launch, not a distinct experiment.

## W&B

- Run name: `confused-butterfly-9`
- Run id: `m30jxiye`
- State: killed; **0s** compute; only `_runtime:0` logged
- Config: k=4, λ_slot=0.25, λ_cov=0.0027, 5000 steps (the serene-cloud-8 slot config)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/m30jxiye

## Parent

[investigation_003](../DESCRIPTION.md)

## Note

Config recovered from W&B (no chat command preserved). Discard for metrics — it produced
none. Its only informational value is dating the slot-run launch sequence.
