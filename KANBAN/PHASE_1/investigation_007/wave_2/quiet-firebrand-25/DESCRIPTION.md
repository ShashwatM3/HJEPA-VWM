# quiet-firebrand-25 — combined "all bigger" (λ=0.2 · 512×2 · n_c=64)

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 5 of 5 (combined interaction run — last of Wave 2)
**Prev:** [helpful-snow-25](../helpful-snow-25/) (weight saturation) · **Next:** — (end of the planned sweep)
**W&B:** `bbrrydax` (attempt-1, **DELETED** — dead) · new run name/id TBD on re-run
**Status:** 🔁 RE-RUNNING on **GPU 3** of the 4-GPU wave — **kept** as the wave's one
interaction-effect check (it took the slot freed by dropping λ=1.0). Attempt 1 died at step 200
(whole-pod death, no data). Launch: [`../../GUIDE.md`](../../GUIDE.md) §3c.

## Hypothesis

The only non-OFAT run in the sweep: it stacks a moderate dose of **all three** levers at once
(`lambda_recon`=0.2, decoder 512×2, `n_c`=64) to test whether they **compound** — i.e. whether the
floor drops more than any single axis alone. The "best shot at the floor" run.

**Prediction (Wave 1):** floor ~0.575–0.585, no break — because Wave 1 showed each axis is
independently flat and the limit is a *shared* mechanism (utilization of `d_c`), so the levers are
not expected to compound. Confounded by design (3 changes at once), so strictly less informative than
the clean single-axis runs.

## Config delta (vs baseline)

`lambda_recon` 0.05→**0.2**, `decoder_dim` 256→**512**, `n_c` 32→**64** (all at once; architecture
change). `lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.2 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 512 --decoder-blocks 2 --n-c 64 \
  --checkpoint-dir /workspace/ckpt/L0.2_D512x2_nc64 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md).
