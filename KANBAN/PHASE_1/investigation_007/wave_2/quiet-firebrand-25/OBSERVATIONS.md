# Observations — quiet-firebrand-25 (combined "all bigger")

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **347 s**, last heartbeat **03:47:43Z** — same instant as all siblings
(synchronized whole-pod death; forensics in [Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0
(initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.045 | untrained (highest init of the wave — init variance, not signal) |
| `coarse_vs_copy_ratio` | 63.0 | random init |
| `c_effective_rank` | 8.82 | pre-training |
| `c_cross_video_cosine` | 0.736 | collapsed random init |

Nothing here is a result.

## What we still don't know

Whether the three levers compound (predicted: no — floor ~0.575–0.585, no break). Because this run is
confounded by design and its outcome is the most predictable in the wave, it is the **lowest re-run
priority** — see [NEXT_STEPS.md](NEXT_STEPS.md). The clean single-axis n_c runs
([pious-mountain-28](../pious-mountain-28/), [earnest-dragon-25](../earnest-dragon-25/)) carry the
decision; this run would only ever be a corroborating "best shot."
