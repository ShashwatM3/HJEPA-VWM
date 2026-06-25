# Observations — peachy-terrain-5

## Key numbers (diagnostic batches, every 500 steps — verified vs W&B `1chv2608`)

| Step | `c_effective_rank` | `coarse_vs_copy_ratio` | `grad_norm` (pre-clip) | Notes |
|---|---|---|---|---|
| 0 | 9.00 | 171.6 | 0.36 | random init |
| 500 | 5.29 | 3.27 | 0.51 | first diag |
| 1000 | 10.48 | 12.93 | 2.44 | rank peak (noise) |
| 1500 | 10.60 | 10.15 | 11.67 | best rank of run; clip already engaging |
| 4500 | 4.45 | 2.76 | **135** | large pre-clip spike, clipped to 1.0 |
| 6000 | 3.95 | 4.24 | **166** | back-half rank settles ~3–5 |
| 7000 | 2.76 | 3.07 | 23.4 | lowest rank logged |
| 8500 | 4.97 | **2.29** | 2.69 | best copy ratio of run |
| 9500 | 5.30 | 7.90 | **47.2** | ratio regressed (prior KANBAN said `0.97` — wrong) |
| 10000 | 4.70 | 7.56 | **176** | warmup ends, `lr_mult=1.0` |
| 10500 | 4.90 | 7.94 | 3.72 | last clean diag |
| 10550 | — | — | **864** | explosion begins (per-step log) |
| 10650 | — | — | **2.6×10⁸** | snowball |
| ~10750 | — | — | NaN | weights corrupted; process died ~10950 on NaN covariance |

Also: `c_std_mean` healthy (~0.74–0.86), `c_dead_dim_frac=0` throughout the healthy
phase. `grad_global_norm` (post-clip) stayed pinned at ≈1.0 the entire run — the
textbook signature of a pre-clip norm that keeps blowing past the clip threshold.

## What broke — the bf16/clip death sequence

The trigger was **not** one freak gradient: large pre-clip spikes (135, 166, 176)
recurred through the whole back half and `grad_clip=1.0` absorbed them. The fatal
interaction came once `lr_mult` reached peak. At step ~10550 a batch produced a
pre-clip norm of **864**; clipping that to norm 1.0 means multiplying every gradient
entry by `1/864 ≈ 0.00116` **in bf16**, which carries only ~2 digits of precision.
The "clipped" gradient was therefore numerical noise pointing in a near-random
direction. The optimizer applied it, the model broke slightly, the next gradient was
bigger (406 → 2.6×10⁸ → 3.5×10¹⁶), and the snowball reached NaN within ~200 steps.
The diagnostics process then crashed running `eigvalsh` on a NaN covariance matrix.
(Full walk-through: chat "death sequence", `cursor_messages` ~4125–4170.)

## What worked

Pipeline, W&B, EMA, frozen encoder, and the variance floor all functioned until the
explosion. Copy ratio showed a real downward trend early (12.93 → 2.29) and rank
briefly reached ~10.6 (step 1500) — proof the stack *can* learn, just not stably to
completion under this schedule.

## Interpretation

Two **independent** problems surfaced: (1) training dynamics unstable at peak LR with
loose clip in bf16 — a numerical/optimizer problem, fixed in code (see
[`NEXT_STEPS.md`](NEXT_STEPS.md)); (2) representational collapse
(`c_effective_rank` ~3–5 against a >60 target) visible the *entire* run, including the
healthy phase — an architectural/regularization problem the stability fixes do **not**
touch. Separating these two is what spawned [investigation_003](../../investigation_003/).
