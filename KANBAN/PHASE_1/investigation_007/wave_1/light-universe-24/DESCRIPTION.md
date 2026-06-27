# light-universe-24 — weight axis, λ_recon = 0.5 (aggressive)

**Wave:** [Wave 1](../DESCRIPTION.md) · **Position:** 3 of 5 (weight ladder, top rung)
**Prev:** [jolly-glade-20](../jolly-glade-20/) (λ=0.2) · **Next:** [gallant-dew-22](../gallant-dew-22/) (decoder axis begins)
**W&B:** `rju7xsh2` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/rju7xsh2
**Status:** COMPLETE (cut ~9.25k, floor plateaued)

## Hypothesis

The aggressive end of the weight ladder — 10× the baseline, recon now = ½ of `L_flow`'s weight.
This is the weight axis's *best shot* at the floor: if reconstruction is weight-bound at all, the
strongest pressure short of recon dominating the loss should show it here. The SWEEP_PLAN also flags
this as the run where the **recon-vs-prediction trade-off** (`L_flow` degrading) might first appear.

## Config delta (vs baseline)

`lambda_recon`: **0.05 → 0.5** (10×). Decoder 256×2, `n_c`=32, `lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.5 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L0.5_D256x2_nc32 --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
