# eager-plant-22 — decoder axis, 512×4 (depth)

**Wave:** [Wave 1](../DESCRIPTION.md) · **Position:** 5 of 5 (decoder ladder, depth — last of Wave 1)
**Prev:** [gallant-dew-22](../gallant-dew-22/) (decoder width) · **Next:** [Wave 2](../../wave_2/DESCRIPTION.md) (latent axis)
**W&B:** `591mt31k` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/591mt31k
**Status:** COMPLETE (cut ~9.1k, floor plateaued)

## Hypothesis

The deepest/largest decoder in the sweep (512 dim × 4 blocks, ~6× baseline params) and the
**strongest probe of the decoder-bound hypothesis**. Depth tends to dominate width for expansion
capacity, so if any decoder config can drop the floor, this is it. This run is also the cleanest
showcase of the **reconstruction-blind-to-prediction** finding (it has the largest `chat − cplus`
gap in the wave) and is the one used in the W&B report's blindness panel.

## Config delta (vs baseline)

`decoder_dim`: **256 → 512**, `decoder_blocks`: **2 → 4**. `lambda_recon`=0.05, `n_c`=32,
`lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/L0.05_D512x4_nc32 --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
