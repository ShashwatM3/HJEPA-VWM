# GUIDE — Run 68, unwhitened EGO4D SigLIP2-B M=512, no variance/covariance

Run this only after the repository is synchronized to the published commit and RunPod W&B login
is already valid. This is a fresh run: do not resume Run 67 and do not use whitening statistics.

## Exact identity

```text
data=ego4d, steps=15000, seed=42, batch-size=64, horizon-k=12
encoder=siglip2_vitb16, revision=3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
encoder-precision=bf16, frame-microbatch=8, attention=sdpa
present-recon-only, lambda-recon=1, lambda-recon-pred=0, recon-loss-mode=cosine
recon-warmup-steps=2000, lambda-var=0, lambda-cov=0, lambda-sigreg=0, lambda-slot=0
whiten-features=false, bottleneck-latent-blocks=3, bottleneck-mixer-dim=512
decoder-dim=512, decoder-blocks=4, n-c=32, lr-bottleneck=1e-4
lr-coarse-flow=1e-4, lr-decoder=1e-4, log-every=50, diag-every=500
```

## Gates

Run the pinned adapter smoke first. Then run exact Stage 0 and resource preflight with the same
flags and unique preflight paths. Stage 0 must show present-only, unwhitened mode, zero flow and
predicted reconstruction, exactly zero variance/covariance terms, finite reconstruction/gradients,
and no skipped update. Resource preflight must pass physical batch 64 with finite values, positive
throughput, and non-null CUDA peak memory. Do not start paid work if any gate fails.

## Paid launch contract

Use detached tmux session `inv016_siglip2_m512_no_geom`, `CUDA_VISIBLE_DEVICES=0`,
`PYTHONUNBUFFERED=1`, `--require-wandb`, and this identity:

```text
wandb entity=smahalanobis-uc-davis
wandb project=hjepa-vwm
wandb group=inv016_unwhitened_m512_geometry
wandb name="Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 no covariance plus variance"
log=/workspace/logs/inv016_unwhitened_siglip2_m512_no_geometry_regularizers.log
checkpoint=/workspace/ckpt/inv016_unwhitened_siglip2_m512_no_geometry_regularizers
provenance=/workspace/ckpt/inv016_unwhitened_siglip2_m512_no_geometry_regularizers/run_provenance.json
```

Append the repository's standard `train.py` launch flags, the W&B flags above, and the resource
paths from `DESCRIPTION.md`; include `--require-wandb`. Verify PID, first step, first diagnostic,
resolved config, W&B ID/name/group, GPU ownership, and output isolation immediately after launch.

## Completion

Completion requires process exit zero, final step 15,000 in the log and W&B history, W&B state
`finished`, final checkpoint and SHA-256, resolved provenance, and zero tripwires. Record all of
that in `OBSERVATIONS.md` and update `ANALYSIS.md` with the measured interpretation.
