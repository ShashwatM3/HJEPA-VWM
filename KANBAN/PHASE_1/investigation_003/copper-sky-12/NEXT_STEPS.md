# Next steps — copper-sky-12

## Why

Stopped ~step 4300: slot loss bites mechanically but **Goodhart** — rank could rise
while `coarse_vs_copy_ratio` stayed bad (~5–11), cross-video cosine ~0.7+. Grad
spikes. Slot path **rejected**.

Real binding issue identified: `lambda_var=0.1` too weak (`c_std_mean` ~0.45 vs
target 1.0). `--lambda-var` CLI added.

## Spawned

**Next run:** [`cerulean-snow-13`](../cerulean-snow-13/) (BRIEF Run 6) — clean
single-variable test: strong variance floor only, no slot/cov.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

**Watch:** `c_std_mean` → 1.0, `c_cross_video_cosine` fall, `coarse_vs_copy_ratio`
< 1, zero `grad_skipped`.
