# pious-mountain-28 — latent axis, n_c = 64 (latent 2×)

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 1 of 5 (latent ladder, step 1 — the core experiment)
**Prev:** [Wave 1 / eager-plant-22](../../wave_1/eager-plant-22/) · **Next:** [classic-yogurt-29](../classic-yogurt-29/) (n_c=128)
**W&B:** `7u5zkw6t` (attempt-1, **DELETED** — dead) · new run name/id TBD on re-run
**Status:** 🔁 RE-RUNNING on **GPU 0** of the 4-GPU wave. Attempt 1 died at step 200 (whole-pod
death, no data). TIER 0 — a core latent bookend. Launch: [`../../GUIDE.md`](../../GUIDE.md) §3c.

## Hypothesis (the one Wave 1 left untested)

The first rung of the latent axis — the **only** binding-constraint candidate Wave 1 didn't
eliminate. Doubling the slot count (32 → 64) doubles `c`'s bandwidth (8,192 → 16,384 numbers, 128:1
→ 64:1). If the 0.585 floor is **latent-capacity-bound** (the project's lead hypothesis, "it's `c`,
not `D`"), this is where the floor should first move *with* a rising `c_effective_rank` and a falling
`coarse_vs_copy_ratio`.

**Wave-1 prior against it:** the under-used axis is `d_c` (per-slot dim, rank ~13/256), which adding
*slots* does not address — so the more likely outcome is floor flat or only cosmetically lower. This
run + [earnest-dragon-25](../earnest-dragon-25/) (n_c=256) are the two that decide it.

## Config delta (vs baseline)

`n_c`: **32 → 64** (architecture change — not resume-compatible). `lambda_recon`=0.05, decoder 256×2,
`lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 64 \
  --checkpoint-dir /workspace/ckpt/L0.05_D256x2_nc64 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md). Re-run plan → [NEXT_STEPS.md](NEXT_STEPS.md).
