# jolly-glade-20 — weight axis, λ_recon = 0.2

**Wave:** [Wave 1](../DESCRIPTION.md) · **Position:** 2 of 5 (weight ladder, step 2)
**Prev:** [toasty-donkey-21](../toasty-donkey-21/) (λ=0.1) · **Next:** [light-universe-24](../light-universe-24/) (λ=0.5)
**W&B:** `5x7aoxnn` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/5x7aoxnn
**Status:** COMPLETE (cut ~9.05k, floor plateaued)

## Hypothesis

The middle rung of the weight ladder (4× the baseline weight). If reconstruction is weight-bound,
the floor should fall monotonically and measurably between [toasty-donkey-21](../toasty-donkey-21/)
(λ=0.1) and [light-universe-24](../light-universe-24/) (λ=0.5). This run tests the *slope* of the
weight axis, not just its endpoints.

## Config delta (vs baseline)

`lambda_recon`: **0.05 → 0.2** (4×). Decoder 256×2, `n_c`=32, `lambda_recon_pred`=0, rest at baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.2 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L0.2_D256x2_nc32 --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
