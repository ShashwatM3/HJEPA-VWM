# Run — royal-cherry-17

## What this run tests

**Resume from pre-spike checkpoint** with **adaptive gradient clipping (AGC)** enabled
(commit `1ae2e09`). Validates that per-tensor AGC on `B` and `F_c` prevents the
step-8500 grad-skip death spiral seen in [`elated-snowflake-15`](../elated-snowflake-15/)
without relying on halved coarse-flow LR alone.

## Hypothesis

AGC (λ_Fc=0.10, λ_B=0.20) clips the 30–100 grad band in-place; post-AGC
`grad_skip_threshold=150` skips only tail catastrophes. Training should pass step 8500
with `grad_skipped=0` and continue updating weights through 15k.

## Command

```bash
cd /workspace/hierarchal-jepa-flow-world-model   # or repo root on pod
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --log-every 50 \
  --diag-every 500
```

AGC is **on by default** (`cfg.train.agc_enabled=True`). No extra flags unless tuning.

## Config delta vs drawn-elevator-16

| Knob | drawn-elevator-16 | This run |
|---|---|---|
| AGC | off (pre-`1ae2e09`) | **on** (λ_B=0.20, λ_Fc=0.10) |
| `grad_skip_threshold` | 50 | **150** |
| `--lr-coarse-flow` | 1e-4 (halved) | default **2e-4** (AGC is the fix) |
| Resume ckpt | `phase1_step7500.pt` | same |

## W&B

- Run name: `royal-cherry-17`
- Run id: `0xv4upvb`
- Runtime: ~4h 52m (killed at step 11350)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0xv4upvb

## Parent

[investigation_005](../DESCRIPTION.md)

## Watch / abort

- **Watch:** `grad_norm`, `grad_skipped`, `agc_Fc_any_clipped`, `agc_Fc_max_ratio`,
  `instability_warn`, `coarse_vs_copy_ratio`, `c_effective_rank`
- **Abort** if `grad_skipped=1` on >10% of steps over any 500-step window
