# Observations — classic-yogurt-29 (n_c=128)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **381 s** (the longest-surviving of the five, by seconds), last heartbeat
**03:47:43Z** — same instant as all siblings (synchronized whole-pod death; forensics in
[Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0 (initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.028 | untrained |
| `coarse_vs_copy_ratio` | 66.8 | random init |
| `c_effective_rank` | 9.64 | pre-training |
| `c_slot_diversity_rank` | 29.6 | trivially higher (more slots) — not a result |
| `c_cross_video_cosine` | 0.735 | collapsed random init |

Indistinguishable from any run's step 0; says nothing about n_c=128.

## What we still don't know

Whether the floor responds monotonically across the latent ladder, and whether n_c=128 trips
slot-collapse. Untested. As an *interpolation* point between the two decisive bookends (n_c=64,
n_c=256), it is lower re-run priority — see [NEXT_STEPS.md](NEXT_STEPS.md).
