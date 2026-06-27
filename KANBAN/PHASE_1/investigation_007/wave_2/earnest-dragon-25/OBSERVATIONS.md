# Observations — earnest-dragon-25 (n_c=256)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **352 s**, last heartbeat **03:47:43Z** — same instant as all siblings
(synchronized whole-pod death; forensics in [Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0
(initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.032 | untrained |
| `coarse_vs_copy_ratio` | 47.2 | random init (slightly lower than siblings — init variance, not signal) |
| `c_effective_rank` | 9.41 | pre-training |
| `c_slot_diversity_rank` | 31.8 | trivially highest (256 slots) — not a result |
| `c_cross_video_cosine` | 0.688 | collapsed random init |

The lower init copy-ratio (47 vs ~63) and higher slot-diversity are trivial consequences of having
8× the slots at random init — **not** evidence of anything.

## What we still don't know — and why this run matters most

This was the **highest-value run of the wave**: the decisive test of the latent-capacity hypothesis
at 8× bandwidth. Its result (floor + `c_effective_rank` + `coarse_vs_copy_ratio` together) is what
would either confirm "latent capacity is the lever" (→ Stage 2) or, far more likely per
[Wave 1](../../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md), confirm the pivot. **Untested.** Tier-0 re-run
priority — see [NEXT_STEPS.md](NEXT_STEPS.md).
