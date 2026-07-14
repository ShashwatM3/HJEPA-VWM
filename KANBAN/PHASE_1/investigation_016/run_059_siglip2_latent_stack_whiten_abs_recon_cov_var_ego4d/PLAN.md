# Plan — run 059 SigLIP 2 EGO4D encoder-substrate control

## Scope

Run the completed pluggable pipeline with `siglip2_vitb16` while reproducing run 058's
present-only, absolute-target, whitened EGO4D recipe on the current strict implementation. No
production-code change is part of this run. DINO access and DINO implementation are not
dependencies.

Run 058 predates the deterministic data/encoder/artifact refactor. It is a historical recipe
reference, not a strict single-delta control. A same-commit V-JEPA companion must be run before
making causal claims about encoder choice.

## Pre-launch work

1. Pull one clean repository commit and install the exact `transformers==4.57.6` lock.
2. Prove the real SigLIP adapter on CUDA: pinned revision, 85,843,200 retained parameters, zero
   trainable encoder parameters, finite `(1,2048,768)` output, and bf16 inference.
3. Revalidate the final full-EGO4D manifest/completeness identity. Do not fit statistics against a
   partial or changed corpus.
4. Resource-preflight the unwhitened exact recipe at physical batch 64. Settle one
   identity-bearing frame microbatch before statistics are computed.
5. Fit exactly 12,800 deterministic EGO4D train clips into a new SigLIP-bound whitening envelope;
   inspect it and record its SHA-256.
6. Run the whitening-active resource preflight, synthetic Stage 0, exact no-step provenance, and a
   100-step full-EGO4D W&B smoke from scratch.
7. Launch 15,000 steps from scratch only after every gate passes.

## Matched recipe

The complete command is in [GUIDE.md](GUIDE.md). Its objective, trainable architecture,
hyperparameters, seed, and full-EGO4D intent match
[run 058](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/). Encoder-owned
normalization, feature layout/dimension, revision, frame microbatch, and whitening identity follow
the SigLIP selection. Intervening implementation/data-order changes prevent treating the
historical run as a one-delta control.

Physical batch 64 is part of the control. Lowering only the encoder frame microbatch is allowed
before statistics are fit because it changes execution chunking, not the selected token values,
but it remains fingerprinted and must match all artifacts. If the full trainable graph cannot fit
physical batch 64 even at frame microbatch 1, stop: lowering the physical batch would add an
optimizer/data-order intervention and requires a revised plan rather than a silent launch.

## Required invariants

- raw input remains float RGB `[0,1]`, `(B,8,3,256,256)`;
- adapter applies SigLIP normalization exactly once and emits all 8x16x16 patch tokens;
- no pooling, temporal subsampling, text tower, V-JEPA stats, or DINO code enters the run;
- trainable output geometry remains `(B,32,256)`;
- `present_recon_only=true`, `recon_residual_target=false`, prediction losses zero;
- loss formulas, weights, warmups, LR schedules, AGC, gradient routing, and seed remain unchanged;
- full EGO4D identity reports `stage-4d-verified`;
- stats, provenance, checkpoint, W&B config, and feature fingerprint all name SigLIP's immutable
  revision;
- final training starts without `--resume` and writes to a run-specific checkpoint directory.

## Abort gates

Do not launch the paid run if any of these occurs:

- repository worktree is dirty or dependency version differs from 4.57.6;
- adapter smoke has the wrong SHA, parameter count, shape, dtype contract, trainable parameters,
  or non-finite output;
- EGO4D completeness validation fails;
- batch 64 is not safe with headroom;
- stats were built with a different encoder fingerprint, frame microbatch, precision, seed,
  dataset fingerprint, or clip count;
- whitening-active resource preflight, Stage 0, exact provenance, or 100-step smoke fails;
- W&B strict initialization/logging fails;
- the run attempts to reuse or resume run 058 state.

## Evidence to retain

- adapter-smoke JSON;
- unwhitened and whitened resource-preflight JSON;
- whitening envelope, inspection output, and SHA-256;
- exact no-step provenance JSON;
- 100-step smoke W&B URL/checkpoint checksum;
- final W&B URL, final checkpoint path/checksum, console log, and run provenance;
- optional pre-training SigLIP detailed-feature rank report for later encoder-relative analysis.

## Verdict method

Use [Reading Cycle B](../../../../GUIDES/READING_EXPERIMENTS.md) because prediction is inactive.
First require clean stability and correct wiring. Then inspect:

- `c_effective_rank`, `c_std_mean`, `c_cross_video_cosine`, `c_dead_dim_frac`;
- `c_slot_diversity_rank_centered` (not raw slot rank alone);
- `L_recon_present`, `L_recon_shuffled_c`, and `L_recon_video_gap`;
- the conditioned share of reconstruction improvement;
- `grad_norm`, skipped/NaN steps, and AGC diagnostics.

Compare trajectories with run 058 as historical context, but do not compare raw reconstruction
magnitudes as if the target spaces or execution commits were identical. Success for run 059
requires both healthy abstract geometry and a materially video-conditioned reconstruction. A
stable low loss with a near-zero video gap is the same template-shortcut failure, not a win.
Encoder-causal comparison waits for the same-commit V-JEPA companion.
