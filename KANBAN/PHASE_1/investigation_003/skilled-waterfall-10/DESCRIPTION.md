# Run — skilled-waterfall-10

**BRIEF "Run 4."** Intended as the *harder-horizon + mild-slot* run — but it executed
at **k=4**, and it ran on the **pre-centering (raw)** slot loss, so it is **not** a
clean test of either lever. Its lasting value was exposing the slot loss/metric
mismatch.

## What this run tested (intent vs reality)

**Intent:** harder horizon (k=12) + gentle slot loss (0.05) — does a harder prediction
task plus a mild slot penalty help, at a weight well below serene's Goodharting 0.25?

**Reality (W&B config `27i1r9qi`): `horizon_k=4`, not 12.** Despite the proposed
command below carrying `--horizon-k 12`, the launched config records `horizon_k=4`.
The `--horizon-k` flag (`183fbc8`) landed only ~4 min before this run started
(06-16 18:12 UTC vs run 18:16 UTC), so the flag was either not pulled or not passed —
a proposed-vs-actual command drift. **Treat this as a k=4 run.**

## Command (proposed — actual run was k=4)

```bash
# proposed:
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
# actual config: horizon_k=4, lambda_slot=0.05, lambda_cov=0.0027, lambda_var=0.10
```

## Config delta vs serene-cloud-8

- `lambda_slot`: 0.25 → **0.05**
- `horizon_k`: **stayed 4** (intended 12; see above)
- `lambda_cov=0.0027` (active), `lambda_var=0.10` (default)
- Slot loss is still the **raw** (pre-centering) formulation — `ffc33ed` had not landed

## W&B

- Run name: `skilled-waterfall-10`
- Run id: `27i1r9qi`
- State: crashed at `_step=2550` (~1h8m); SSH drop noted ~step 950 in chat; grad-skips
  from step 1500 on (grad_norm 55 → 204)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/27i1r9qi

## Parent

[investigation_003](../DESCRIPTION.md)
