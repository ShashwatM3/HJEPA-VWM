# helpful-snow-25 — weight saturation, λ_recon = 1.0

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 4 of 5 (weight-axis saturation extreme)
**Prev:** [earnest-dragon-25](../earnest-dragon-25/) (n_c=256) · **Next:** [quiet-firebrand-25](../quiet-firebrand-25/) (combined)
**Extends:** the Wave-1 weight ladder ([light-universe-24](../../wave_1/light-universe-24/), λ=0.5)
**W&B:** `tw685b5g` (attempt-1, **DELETED** from W&B — dead link)
**Status:** ❌ DROPPED from the 4-GPU re-run. Attempt 1 died at step 200 (whole-pod death); the
re-run has only 4 GPUs, and this λ=1.0 weight-saturation run is the cut — see below / [Wave 2
DESCRIPTION](../DESCRIPTION.md).

## Hypothesis

The saturation extreme of the **weight** axis: λ_recon = 1.0 makes reconstruction *equal* to
`L_flow`'s weight (20× the baseline). Closes the OFAT blind spot for the weight ladder — confirms
whether the floor stays flat at the extreme or whether the faint Wave-1 trend suddenly bites. Also
the run where `L_flow` degradation (recon over-competing with prediction) should be most visible.

**Prediction (Wave 1):** floor ~0.581 (extrapolated from the −0.005/doubling slope), no break, with
`L_flow` ticking up. The most predictable run in the wave.

## Forensic significance

This run is the **architectural control** for the whole failed wave: it is identical in architecture
to Wave 1 (n_c=32, decoder 256×2) — only the weight differs. Because it **died at the same step 200
as the n_c=256 run**, it proves the Wave-2 failure was *not* an `n_c` shape bug but a whole-pod event.

## Config delta (vs baseline)

`lambda_recon`: **0.05 → 1.0** (20×). Decoder 256×2, `n_c`=32, `lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L1.0_D256x2_nc32 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md).
