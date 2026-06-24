# Run — royal-cherry-17

## What this run tested

**Fresh full-SSv2 Phase 1 run from step 0** with **adaptive gradient clipping (AGC)**
enabled (commit `1ae2e09`). Same cerulean config as [`elated-snowflake-15`](../elated-snowflake-15/)
(`horizon_k=12`, `lambda_var=0.5`, peak LRs 1e-4 / 2e-4, seed 42) plus AGC knobs.

> **Note:** KANBAN originally described a resume from `phase1_step7500.pt`. W&B history
> shows steps **0–11350** with warmup `lr_mult` from 0 — this was a **fresh run**, not a
> resume. See `OBSERVATIONS.md` §Run identity.

## Hypothesis

Per-tensor AGC (λ_B=0.20, λ_Fc=0.10) clips moderate `F_c` backward spikes; post-AGC
`grad_skip_threshold=150` skips only tail catastrophes. Training should pass the
step-8500 break that killed elated and complete 15k with learning intact.

## Command (actual — fresh, per W&B)

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --log-every 50 \
  --diag-every 500
```

AGC on by default after `1ae2e09`. No `--resume` in logged config.

## Config (W&B `0xv4upvb`)

| Knob | Value |
|---|---|
| `agc_enabled` | True |
| `agc_lambda_bottleneck` | 0.20 |
| `agc_lambda_coarse_flow` | 0.10 |
| `agc_eps` | 1e-3 |
| `grad_skip_threshold` | 150 |
| `grad_clip` | 0.5 |
| `instability_warn` | grad > 30 ∧ L_flow > 1.0 |
| `lr_coarse_flow` | 2e-4 |
| `lr_bottleneck` | 1e-4 |
| `lambda_var` | 0.5 |
| `horizon_k` | 12 |
| `seed` | 42 |

## W&B

- Run name: `royal-cherry-17`
- Run id: `0xv4upvb`
- State: **killed** at step 11350 (target 15000)
- Runtime: ~4h 52m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0xv4upvb

## Parent

[investigation_005](../DESCRIPTION.md)
