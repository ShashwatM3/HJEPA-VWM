# Run — sleek-leaf-7

**BRIEF Run 2 / VICReg Run A.** First full-SSv2 baseline after P1 diagnostic.

## What this run tested

Init fixes + full SSv2, no slot/cov active (`lambda_cov=0` logs `L_cov` only).
Isolated: does init + full data fix rank ~5? Stopped ~step 3500 for data-confound verdict.

## Command (confirmed — chat `cursor_messages` ~5832/5867; W&B config `rpxyg9qt`)

```bash
python train.py --data ssv2 --steps 15000 --log-every 50 --diag-every 500
```

Post-retune defaults; init fixes baked in (`62b94dd`); per-head entropy fix (`a96c0d6`).
W&B config confirms `lambda_var=0.10`, `horizon_k=4`, `lambda_cov=0`, `lambda_slot` unset
(slot loss `4418410` had not landed yet — this is a clean no-regularizer baseline). Run
stopped ~step 3500 for the data-confound verdict (W&B shows it reached `_step=4450`).

## Config delta vs peachy-terrain-5

- Full SSv2 (not tiny)
- Halved LRs, 1.5k warmup, 15k steps, grad clip 0.5, skip guard
- Per-head `c_attn_entropy_min` active (post `a96c0d6`)

## W&B

- Run name: `sleek-leaf-7`
- Run id: `rpxyg9qt`
- Runtime: ~1h 39m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/rpxyg9qt

## Parent

[investigation_003](../DESCRIPTION.md) — also [investigation_004 Run A](../investigation_004/)
