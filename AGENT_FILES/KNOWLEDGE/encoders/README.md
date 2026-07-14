# Encoder Knowledge

This directory contains the frozen-encoder research and the implementation/experiment
plan for making HJEPA-VWM encoder-pluggable.

## Recommended first pair

1. [`DINOv3_ViT-B-16.md`](DINOv3_ViT-B-16.md) — selected DINO encoder, exact feature
   contract, quality/weight rationale, temporal trade-offs, license, and preflight gates.
2. [`SigLIP_2_ViT-B-16.md`](SigLIP_2_ViT-B-16.md) — selected non-DINO control, exact
   feature contract, preprocessing differences, rationale, and preflight gates.
3. [`ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md`](ENCODER_PLUGGABILITY_AND_PARALLEL_EXPERIMENT_PLAN.md)
   — complete code-change design, test matrix, migration sequence, paired run recipe,
   analysis rules, full-prediction follow-up, and user actions.
4. [`GUIDE_encoders.md`](GUIDE_encoders.md) — dependency-segmented human/coding-agent execution guide:
   DINO-independent common work, separate SigLIP/DINO lanes, a final join gate, manual
   access/EGO4D coordination, copy/paste implementation prompts, whitening/resource
   preflights, concurrent-or-sequential launch blocks, and full-mode follow-up.

The first pair is intentionally shape-matched: both frame encoders produce an
8x16x16 lattice of 768-dimensional patch tokens at 256px. Neither is temporally aware, so
present-only reconstruction is a substrate test; a later full-prediction control is needed
to evaluate the loss of V-JEPA2's tubelet-level temporal modeling.

Implementation status (2026-07-14): the encoder-independent data/model/training,
determinism/checkpoint, whitening/rank/drift, artifact, and preflight stages are complete.
`encoders.py` provides pinned real V-JEPA2 and SigLIP 2 adapters on
`transformers==4.57.6`; SigLIP's vision-only 8x16x16x768 path has passed a real Mac/MPS
smoke. Dataset identities include container frame counts, exact resume consumes the saved
sampler position, resource reports include CUDA-event examples/frames/tokens rates, and
frame encoders receive pre-concatenation frame rank/norm diagnostics. DINO remains the only
deliberately unresolved alias until its separate offline and gated-real lane pins and
validates an immutable snapshot. CUDA, real SSv2/EGO4D, stats, resource, and paid-run
evidence must be produced on the authenticated RunPod using `GUIDE_encoders.md`.
