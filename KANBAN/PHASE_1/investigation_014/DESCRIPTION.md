# Investigation 014 - V-JEPA embedding rank budget

## Status

OPEN

## Question

What is the intrinsic effective-rank structure of the frozen V-JEPA `e` embeddings on the fixed
Phase 1 probe set, before any trainable bottleneck `B` or abstract latent `c_t` is involved?

## Context

Phase 1 repeatedly reads `c_effective_rank` as a representation-health signal, but until this
probe there was no same-formula reference for the frozen encoder side. The rank probe measures
V-JEPA `e` directly with the same covariance entropy mechanism used by `diagnostics.effective_rank`.

This investigation records the first measured encoder-side rank budget:

```bash
python rank_probe.py --data ssv2 --probe-videos 64
```

Probe definition:

- Dataset: `ssv2`
- Split: `validation`
- Probe videos: `64`
- Seed: `42`
- Window: anchor context window only
- Encoder tokens per video: `N_ctx = 1024`
- Encoder feature dim: `D_e = 1024`
- Result JSON: `logs/drift_probe/rank_results_ssv2_validation_n64_seed42.json`

## Why This Matters

The bottleneck does not compress a full-rank, clean 1024-dimensional object. It compresses a frozen
encoder representation that is already anisotropic: useful variance is concentrated in a few
hundred directions, while many dimensions live in a weak long tail.

That means future Phase 1 interpretation should separate:

- loss of genuine encoder-side information during `e -> c` compression;
- removal of low-energy/noisy/redundant encoder directions;
- failure modes where `c_t` becomes lower-rank than the frozen encoder signal actually warrants.
