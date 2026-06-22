# Run — exalted-lion-6

## What this run tested

Plan Phase 04 **P1 diagnostic** (~500 steps on `ssv2_tiny`): confirm new collapse
instrumentation on the real pipeline and reproduce Run 1 rank stagnation at low cost
before implementing VICReg-C.

## Command

```bash
python train.py --data ssv2_tiny --steps 500 --log-every 50 --diag-every 100
```

## Config delta

- Pre–per-head-attention-entropy fix (head-averaged `c_attn_entropy` still ~1.0)
- `lambda_var=0.10`, `horizon_k=4`, no slot/cov
- Post–Run-1 stability retune (halved LRs, skip guard)

## W&B

- Run name: `exalted-lion-6`
- Run id: `wv69n7n5`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/wv69n7n5

## Parent

[investigation_003](../DESCRIPTION.md) — also informs [investigation_004](../investigation_004/)
