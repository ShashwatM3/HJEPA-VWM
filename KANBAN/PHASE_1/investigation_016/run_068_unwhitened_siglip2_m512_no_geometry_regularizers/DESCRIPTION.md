# Run 68 — Investigation 16 · SigLIP2-B M=512 without geometry regularizers

## Status

CRASHED — W&B `j7a3tzj5` terminated at logged step 14,350. All logged optimizer tripwires remained
zero through the last row, so the history does not show a numerical-training failure; W&B does not
identify the external termination cause. This is a partial endpoint, not a completed 15,000-step
run.

## Scientific question

Does the revised bottleneck preserve and use input-dependent SigLIP2 features when the internal
memory width is `M=512`, without whitening and without the variance/covariance losses? Every
scientific and runtime variable is inherited from Run 67 except `lambda_var=0` and `lambda_cov=0`.
This isolates the architectural bottleneck intervention from geometry-regularizer pressure.

## Fixed recipe

- Dataset: full EGO4D; seed `42`; physical batch `64`; horizon `12`.
- Frozen encoder: implemented standard-ViT lane `siglip2_vitb16`, pinned revision
  `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`; output `(B,2048,768)`.
- Bottleneck: three latent refinement blocks, mixer/internal memory width `512`, `N_c=32`,
  external slot width `D_c=256`, with the final channel projection at the bottleneck output.
- Decoder: dimension `512`, four blocks.
- Objective: present-only raw-feature cosine reconstruction, `lambda_recon=1`, warmup `2000`,
  no predicted reconstruction, flow, residual target, whitening, SIGReg, or slot loss.
- Geometry regularizers: `lambda_var=0`, `lambda_cov=0`.
- Schedule: `15,000` steps, learning rates `1e-4` for bottleneck/coarse-flow/decoder, bf16
  encoder, frame microbatch `8`, SDPA attention, logging every `50`, diagnostics every `500`.

## Interpretation boundary

Run 68's raw reconstruction is in the SigLIP2 feature space and must not be compared numerically
to V-JEPA raw loss as if the targets were identically scaled. The key controls are correct versus
shuffled-code reconstruction, cross-example cosine, effective rank, centered slot rank, code std,
dead dimensions, and stability. The fixed diagnostic batch is within one EGO4D source UID, so its
cosine is a within-source cross-example signal rather than a global cross-source statistic.

## Provenance and output isolation

- Published source commit: `4c402556e0b85684499f5a111630b7c8bc20ca89` on
  `phase1-v0.2-frozen-encoder`.
- Dataset fingerprint: `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c`.
- Encoder feature fingerprint: `2340dea66764e6f8800443add7f02dd5586ae4e017bf81b824121b3256bce848`.
- Log: `/workspace/logs/inv016_unwhitened_siglip2_m512_no_geometry_regularizers.log`.
- Checkpoints: `/workspace/ckpt/inv016_unwhitened_siglip2_m512_no_geometry_regularizers/`.
- Provenance: `/workspace/ckpt/inv016_unwhitened_siglip2_m512_no_geometry_regularizers/run_provenance.json`.
- Preflight report: `/workspace/preflight/inv016_unwhitened_siglip2_m512_no_geometry_regularizers/ego4d_siglip2_m512_no_geometry_regularizers_resource.json`.
- W&B group: `inv016_unwhitened_m512_geometry`.
- W&B name: `Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 no covariance plus variance`.
- W&B ID: `j7a3tzj5`.
- W&B URL: <https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/j7a3tzj5>.
- tmux session: `inv016_siglip2_m512_no_geom`.
- Training PID at launch: `2087961`.

## Gate record

- SigLIP2 CUDA adapter smoke: passed; finite `(1,2048,768)`, zero trainable encoder parameters,
  exact pinned revision, peak encoder memory `414,810,112` bytes.
- Stage 0: passed with present-only/unwhitened mode, zero weighted variance/covariance terms,
  finite losses/gradients, and zero skipped updates. The diagnostic fields `L_var` and `L_cov`
  remain measurable raw terms, but their configured weights are both zero.
- Resource preflight: the new full inventory scan was allowed to progress for ten minutes but
  timed out before emitting its JSON. The existing Run 67 resource report was reused as a valid
  architecture-identical upper bound: batch 64, total peak `19,783,331,840` bytes, throughput
  `59.00` examples/s, finite metrics, and zero skipped updates. The only change is the two loss
  weights, which cannot increase the model's memory footprint.

Run 67 (`ufbeokj2`) is the stopped regularized reference. Run 68 must start from an empty,
unique checkpoint directory and must not resume Run 67.
