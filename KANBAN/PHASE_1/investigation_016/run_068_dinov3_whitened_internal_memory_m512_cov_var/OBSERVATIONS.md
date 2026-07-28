# Observations — Run 68, DINOv3 whitened internal-memory M=512 covariance plus variance

> **Local-label/status correction (2026-07-27):** exact completed evidence now lives under
> canonical local [Run 070](../run_070_whitened_dinov3_m512_cov_var/), W&B `qqozribu`.
> Its W&B-backed analysis supersedes the empty pre-launch placeholder below.

## Status

PLANNED — not yet launched. No metrics recorded yet.

Fill this file after the run completes:

- W&B ID/URL and final state (`finished`).
- DINOv3 whitening-envelope path, payload fingerprint, and file SHA-256.
- Final checkpoint path + SHA-256 and committed provenance artifact.
- Present-only Reading Cycle B table (Q1–Q6).
- Preregistered late-window medians (steps 12,000–14,500): active `L_recon`, correct-code
  `L_recon_present`, shuffled-code `L_recon_shuffled_c`, correct-vs-shuffled gap, exact-chunk
  conditioned share, mean code std, cross-example cosine, effective rank, and slot-diversity rank.
- Within-run diagnostic comparison against unwhitened DINOv3 Run 67 (`it7sq8nz`).

Do not compare whitened and unwhitened raw reconstruction magnitudes as though they share a target
space. Do not invent metric values; pull them from W&B.
