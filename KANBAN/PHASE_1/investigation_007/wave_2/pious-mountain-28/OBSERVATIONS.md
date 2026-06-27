# Observations — pious-mountain-28 (n_c=64)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **350 s**, last heartbeat **03:47:43Z** — the same instant as all four
siblings (synchronized whole-pod death; see [Wave 2 OBSERVATIONS](../OBSERVATIONS.md) for the
forensics). Diagnostics log every 500 steps, so the only logged row is **step 0 (initialization)**.

### The step-0 values are initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.030 | ≈ 1.0 = untrained (as bad as predicting the mean) |
| `coarse_vs_copy_ratio` | 63.0 | random-init value (every run starts here) |
| `c_effective_rank` | 8.82 | pre-training |
| `c_cross_video_cosine` | 0.736 | collapsed random init (drops to ~0.2 only after training) |

These are indistinguishable from every Wave-1 run's step 0 and say **nothing** about the n_c=64
hypothesis. The recon warmup runs to 2000 steps; this died at 10% of warmup.

## What we still don't know

Whether doubling `n_c` moves the 0.585 floor — the central open question of investigation_007. The
[pre-registered prediction](../../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md) (floor flat or cosmetic,
copy-ratio still >1 → pivot) is **untested**. This run is a Tier-0 re-run priority — see
[NEXT_STEPS.md](NEXT_STEPS.md).
