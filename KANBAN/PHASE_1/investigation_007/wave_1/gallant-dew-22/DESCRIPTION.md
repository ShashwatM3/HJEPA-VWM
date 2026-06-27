# gallant-dew-22 — decoder axis, 512×2 (width)

**Wave:** [Wave 1](../DESCRIPTION.md) · **Position:** 4 of 5 (decoder ladder, width)
**Prev:** [light-universe-24](../light-universe-24/) (weight axis ends) · **Next:** [eager-plant-22](../eager-plant-22/) (decoder depth)
**W&B:** `708jrel8` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/708jrel8
**Status:** COMPLETE (last logged ~8.5k, floor plateaued)

## Hypothesis

First of the two decoder runs, holding `lambda_recon` at the baseline 0.05 and *widening* the
decoder (256 → 512 dim, ~3.4× params). Tests the **decoder-bound** hypothesis (the tech-lead's
intuition): that one thin `D` is too small to expand `c → e`, so a wider decoder reconstructs better
and the floor drops. Width is the most direct capacity increase, so it goes first.

## Config delta (vs baseline)

`decoder_dim`: **256 → 512** (blocks held at 2). `lambda_recon`=0.05, `n_c`=32,
`lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 512 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L0.05_D512x2_nc32 --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
