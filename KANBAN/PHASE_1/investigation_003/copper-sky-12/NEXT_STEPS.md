# Next steps — copper-sky-12

## Why

Killed at step 5650: the centered slot loss bites mechanically (slot diversity pinned
~20–26) but it is **Goodhart** — `c_effective_rank` *collapsed* to ~4.8,
`c_cross_video_cosine` rose to **0.84**, `coarse_vs_copy_ratio` stayed bad, and a
4×10⁴ grad spike appeared. Slot path **rejected as a training objective** (across both
copper and [`olive-terrain-11`](../olive-terrain-11/)).

The real binding constraint, now visible in two slot runs: **`lambda_var=0.1` is too
weak** — `c_std_mean` sits at ~0.4–0.5 the whole run, i.e. `L_flow` out-pulls the
variance floor and the latent shrinks/correlates. The hypothesis reframes from "add a
decorrelation/diversity term" to "**the anti-collapse term we already have is
under-dosed**." A `--lambda-var` CLI flag was added (`562de1b`) to test this cleanly,
and the grad-spike question was deliberately *not* fixed yet — the hypothesis was that
spikes are downstream of collapse, so fixing collapse might calm them (it did).

## Spawned

**Next run:** [`cerulean-snow-13`](../cerulean-snow-13/) (BRIEF Run 6) — clean
single-variable test: strong variance floor only, no slot/cov.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

**Watch:** `c_std_mean` → 1.0, `c_cross_video_cosine` fall, `coarse_vs_copy_ratio`
< 1, zero `grad_skipped`.
