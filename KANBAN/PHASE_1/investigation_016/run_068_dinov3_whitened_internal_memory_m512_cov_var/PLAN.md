# Plan — Run 68, DINOv3 whitened internal-memory M=512 covariance plus variance

## Execution state

The run was launched. Exact completion and artifact identities remain pending; the pre-launch plan below is retained as the intended recipe.

## Scope

CLI-only whitening-bundle follow-up to completed DINOv3 Run 67. Preserve the full unwhitened M=512
scientific recipe and enable fixed offline DINOv3/EGO4D whitening with the canonical Investigation
016 geometry weights: `lambda_var=0.5`, `lambda_cov=0.01`, SIGReg and slot loss off.

## No production-code change

Current `train.py`, `whiten_stats.py`, `provenance.py`, and `EncoderSpec` already provide the strict
encoder-bound whitening path. No model or training-code edit is part of this experiment.

## Pre-launch work

1. Verify a clean branch `codex/task3-dino-run066` (or an equivalent branch) whose `HEAD`
   descends from base commit `083cf8a6e87168702efe46ac6bfe485756dcb439`, with every committed
   change since that base restricted to the Run 067/068 KANBAN documentation paths; abort
   validation on any code, configuration, or training-script drift, then pass the repository test
   suite.
2. Prove the real pinned DINOv3 adapter on CUDA and revalidate full EGO4D completeness.
3. Resource-preflight the exact trainable recipe at physical batch 64, first without a whitening
   artifact, to settle frame microbatch 32 or the documented lower ladder.
4. Fit exactly 12,800 deterministic full-EGO4D train clips into a new DINOv3-bound whitening
   envelope; inspect it and record its SHA-256.
5. Strictly validate encoder, revision, feature/preprocessing identity, dataset/split, seed,
   precision, attention implementation, frame microbatch, and clip count.
6. Run whitening-active resource preflight, Stage 0, and exact no-step provenance.
7. Launch 15,000 steps from scratch only after every gate passes.

## Matched recipe

The complete command is in [`GUIDE.md`](GUIDE.md). Relative to Run 67, the only scientific changes
are `lambda_var: 0 -> 0.5`, `lambda_cov: 0 -> 0.01`, and activation of the fixed offline whitening
path at epsilon `1e-4` with an expected 12,800-clip artifact. Operational names and paths change to
avoid collisions. Frame microbatch may move only down the `32/16/8/4/2/1` resource ladder before
statistics are fit; physical batch 64 is locked.

## Required invariants

- raw input remains float RGB `[0,1]`, `(B,8,3,256,256)`;
- DINOv3 normalization is applied exactly once and the selected dense token contract is unchanged;
- stats and training share encoder revision, preprocessing, precision, SDPA, frame microbatch,
  dataset fingerprint, transform seed, and full 12,800-clip budget;
- `present_recon_only=true`, prediction inactive, residual target off;
- `M=512`, three latent blocks, external `(32,256)` code, 512-by-4 decoder;
- final training starts without `--resume` in a fresh run-specific checkpoint directory;
- no V-JEPA or SigLIP whitening artifact is read.

## Abort gates

Do not launch if the worktree is dirty, the base commit is not an ancestor of `HEAD`, any
committed change since the base falls outside the Run 067/068 KANBAN documentation paths, any
code/configuration/training-script drift appears, tests fail, the DINO adapter identity is wrong,
EGO4D completeness fails, physical batch 64 lacks headroom, stats metadata mismatches,
whitening-active preflight or Stage 0 fails, exact provenance differs unexpectedly, W&B strict
initialization fails, or any command attempts to resume the unwhitened checkpoint.

## Evidence to retain

- adapter smoke JSON and resource-preflight JSON;
- whitening envelope, inspection output, payload fingerprint, and file SHA-256;
- exact no-step provenance JSON;
- final W&B ID/URL/state and whitening/provenance artifacts;
- final log, checkpoint path/SHA-256, and run provenance.

## Verdict method

Use present-only Reading Cycle B. Compare stability, code geometry, correct-vs-shuffled separation,
and conditioned share against Run 67. Whitening changes the target coordinates, so raw loss
magnitudes are not directly comparable. The fixed validation batch remains within-source, so no
global cross-source preservation claim is available.
