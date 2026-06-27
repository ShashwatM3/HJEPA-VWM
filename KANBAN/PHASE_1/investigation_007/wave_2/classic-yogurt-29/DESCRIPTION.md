# classic-yogurt-29 — latent axis, n_c = 128 (latent 4×)

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 2 of 5 (latent ladder, step 2)
**Prev:** [pious-mountain-28](../pious-mountain-28/) (n_c=64) · **Next:** [earnest-dragon-25](../earnest-dragon-25/) (n_c=256)
**W&B:** `ryuh8cpr` (attempt-1, **DELETED** — dead) · new run name/id TBD on re-run
**Status:** 🔁 RE-RUNNING on **GPU 1** of the 4-GPU wave (kept — gives the latent *trend*, not just
bookends). Attempt 1 died at step 200 (whole-pod death, no data). Launch:
[`../../GUIDE.md`](../../GUIDE.md) §3c.

## Hypothesis

The middle rung of the latent ladder — 4× the baseline bandwidth (128:1 → 32:1). Tests whether the
floor responds *monotonically* to slot count between n_c=64 and n_c=256. Also the wave's primary
**slot-collapse watch** at the original design stage (more slots = more to keep diverse;
`c_slot_diversity_rank` / `c_cross_video_cosine` were the metrics to monitor).

## Config delta (vs baseline)

`n_c`: **32 → 128** (architecture change). `lambda_recon`=0.05, decoder 256×2, `lambda_recon_pred`=0,
rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 128 \
  --checkpoint-dir /workspace/ckpt/L0.05_D256x2_nc128 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md). Re-run priority → [NEXT_STEPS.md](NEXT_STEPS.md).
