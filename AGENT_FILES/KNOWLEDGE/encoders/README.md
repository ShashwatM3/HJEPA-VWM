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
4. [`GUIDE.md`](GUIDE.md) — ordered human/coding-agent execution guide: manual access and
   EGO4D coordination, five copy/paste implementation prompts, verification gates,
   whitening/resource preflights, exact paired launch blocks, and full-mode follow-up.

The first pair is intentionally shape-matched: both frame encoders produce an
8x16x16 lattice of 768-dimensional patch tokens at 256px. Neither is temporally aware, so
present-only reconstruction is a substrate test; a later full-prediction control is needed
to evaluate the loss of V-JEPA2's tubelet-level temporal modeling.
