# Next steps — Run 67, DINOv3 unwhitened internal-memory M=512

> **Local-label correction (2026-07-27):** current follow-up belongs to canonical local
> [Run 069](../run_069_unwhitened_dinov3_m512_no_geometry_regularizers/) for W&B `it7sq8nz`.
> The list below is preserved source-branch history.

1. Keep this completed arm immutable; its W&B run, committed provenance artifact, and persistent-
   volume final checkpoint are valid.
2. Pull the preregistered steps 12,000–14,500 diagnostic window from W&B and replace only the
   explicit placeholders in [`OBSERVATIONS.md`](OBSERVATIONS.md); do not infer medians from the final
   snapshot.
3. Compare within-run diagnostics against the V-JEPA M=512 control
   ([`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/),
   W&B `4biwq87o`). Do not compare raw cross-encoder cosine losses as though they share a target
   space.
4. Treat source-diverse representation health as a separate diagnostic repair. The fixed batch can
   establish exact-chunk dependence but not global cross-source preservation or collapse.
5. If a whitened DINOv3 follow-up is approved, use a newly computed encoder-bound DINOv3/EGO4D
   whitening artifact; do not reuse V-JEPA or SigLIP whitening statistics.
