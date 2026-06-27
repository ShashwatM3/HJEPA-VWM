# toasty-donkey-21 — weight axis, λ_recon = 0.1

**Wave:** [Wave 1](../DESCRIPTION.md) · **Position:** 1 of 5 (weight ladder, step 1)
**Prev:** baseline [`easy-blaze-19`](../../../investigation_006/easy-blaze-19/) (λ=0.05) · **Next:** [jolly-glade-20](../jolly-glade-20/) (λ=0.2)
**W&B:** `a2trqp9c` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/a2trqp9c
**Status:** COMPLETE (cut ~9.2k, floor plateaued)

## Hypothesis

The first rung of the weight ladder. If the 0.60 floor is **weight-bound** (we under-ask
reconstruction at the 5% baseline), then merely doubling `lambda_recon` 0.05 → 0.1 should already
nudge `L_recon_present` down and lift `c_effective_rank`. This run sets the low anchor of the
weight axis.

## Config delta (vs `easy-blaze-19` baseline)

`lambda_recon`: **0.05 → 0.1** (2×). Everything else at baseline: decoder 256×2, `n_c`=32,
`lambda_recon_pred`=0, horizon-k 12, `lambda_var` 0.5.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.1 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L0.1_D256x2_nc32 --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
