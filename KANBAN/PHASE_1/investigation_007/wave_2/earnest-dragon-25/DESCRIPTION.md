# earnest-dragon-25 — latent axis, n_c = 256 (latent 8×, saturation)

**Wave:** [Wave 2](../DESCRIPTION.md) · **Position:** 3 of 5 (latent ladder, saturation extreme)
**Prev:** [classic-yogurt-29](../classic-yogurt-29/) (n_c=128) · **Next:** [helpful-snow-25](../helpful-snow-25/) (weight saturation)
**W&B:** `2xsd5jwr` (attempt-1, **DELETED** — dead) · new run name/id TBD on re-run
**Status:** 🔁 RE-RUNNING on **GPU 2** of the 4-GPU wave. TIER 0 — highest-value run of the wave.
Attempt 1 died at step 200 (whole-pod death, no data). **Watch for OOM** (most VRAM-hungry, though
OOM isn't expected — no CLI batch flag; lower `global_batch`/`num_workers` in `config.py` if it ever
hits) and slot collapse. Launch: [`../../GUIDE.md`](../../GUIDE.md) §3c.

## Hypothesis

The saturation extreme of the latent axis — **8× the baseline bandwidth** (128:1 → 16:1). This is
the *strongest possible test* of the latent-capacity hypothesis: if the floor is bound by `c`'s
information bandwidth, the effect must be visible here or nowhere. Added during Wave-1 analysis
specifically to close the OFAT blind spot ("flat axis" vs "axis not pushed hard enough"): a flat
floor even at 8× slots is decisive evidence that capacity is *not* the constraint.

**Diagnostic to separate two mechanisms** (per [Wave 1 prediction](../../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md)):
if the floor drops here, check `c_effective_rank` — if rank rises with it, `c` genuinely got richer;
if rank stays ~13 (fraction craters) while the floor only dips, it's a cosmetic decoder-KV-token
effect, not enrichment. Also the **highest slot-collapse risk** run (`c_slot_diversity_rank`,
`c_cross_video_cosine`).

## Config delta (vs baseline)

`n_c`: **32 → 256** (8×; architecture change, most VRAM-hungry of the wave). `lambda_recon`=0.05,
decoder 256×2, `lambda_recon_pred`=0, rest baseline.

## Command

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
  --decoder-dim 256 --decoder-blocks 2 --n-c 256 \
  --checkpoint-dir /workspace/ckpt/L0.05_D256x2_nc256 --log-every 50 --diag-every 500
```

Failure record → [OBSERVATIONS.md](OBSERVATIONS.md). Re-run plan → [NEXT_STEPS.md](NEXT_STEPS.md).
