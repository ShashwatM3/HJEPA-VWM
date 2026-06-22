# Run — exalted-lion-6

## What this run tested

Plan Phase 04 **P1 instrumented diagnostic** (~500 steps on `ssv2_tiny`): confirm
collapse instrumentation on the real pipeline at low cost before the full-SSv2
collapse arc. Init fixes (orthogonal queries, zero-init `out_mlp`) were **baked in**.
No auxiliary regularizers — all at defaults.

## Command

```bash
python train.py --data ssv2_tiny --steps 500 --log-every 50 --diag-every 100
```

(`--lambda-cov`, `--lambda-slot`, and `--lambda-var` all default / off.)

## Config delta

- Init fixes baked into `models.py`
- `lambda_var=0.10` (default), `horizon_k=4`, `lambda_cov=0`, `lambda_slot=0`
- Post–Run-1 stability retune (halved LRs, grad skip guard)
- Head-averaged `c_attn_entropy` still active (per-head fix landed in `a96c0d6`, after this run)

## W&B

- Run name: `exalted-lion-6`
- Run id: `wv69n7n5`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/wv69n7n5

## Parent

[investigation_003](../DESCRIPTION.md) — chronologically first run in this investigation
