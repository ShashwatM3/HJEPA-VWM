# GUIDE — Run 67, unwhitened EGO4D SigLIP2-B M=512 with covariance plus variance

This is Run 66 with only the frozen encoder selection changed to the implemented standard-ViT
lane. It is a fresh 15,000-step run; do not resume another checkpoint and do not use whitening.

## 1. Exact identity

```text
data = ego4d
seed = 42
encoder = siglip2_vitb16
encoder_revision = 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
present_recon_only = true
whiten_features = false
bottleneck_mixer_dim = 512
n_c = 32
d_c = 256
lambda_recon = 1.0
lambda_var = 0.5
lambda_cov = 0.01
lambda_sigreg = 0
lambda_slot = 0
```

W&B:

```text
name  = Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 covariance plus variance
group = inv016_unwhitened_m512_geometry
```

Outputs:

```text
tmux       = inv016_siglip2_m512_cov_var
log        = logs/inv016_unwhitened_siglip2_m512_cov_var.log
checkpoint = /workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var
provenance = /workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var/run_provenance.json
preflight  = /workspace/preflight/inv016_unwhitened_siglip2_m512_cov_var/ego4d_siglip2_m512_cov_var_resource.json
```

## 2. Gates

On the clean published Run-66 commit, require the full test suite/model/diagnostic smokes and W&B
login already recorded for the pod. Then run:

```bash
python encoders.py --smoke \
  --encoder siglip2_vitb16 \
  --revision 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab \
  --precision bf16 --frame-microbatch 8 \
  --attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size 1 --device cuda
```

Require a finite `(1,2048,768)` output, zero trainable encoder parameters, the exact requested and
resolved revision, and a non-null CUDA peak. Run exact `--stage0-only` and `--resource-preflight`
gates with the arguments below. Stage 0 must show present-only/unwhitened mode, zero flow/predicted
reconstruction, finite positive variance/covariance terms, and no skipped update. Resource
preflight must pass physical batch 64 with finite metrics, positive throughput, and non-null peak
memory.

## 3. Shared paid/gate arguments

```bash
--data ego4d
--steps 15000
--seed 42
--encoder siglip2_vitb16
--encoder-revision 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
--encoder-precision bf16
--encoder-frame-microbatch 8
--encoder-attention-implementation sdpa
--hf-cache-dir /workspace/hf_cache
--batch-size 64
--horizon-k 12
--present-recon-only
--lambda-recon 1.0
--lambda-recon-pred 0
--recon-loss-mode cosine
--recon-warmup-steps 2000
--lambda-var 0.5
--lambda-cov 0.01
--lambda-sigreg 0
--lambda-slot 0
--bottleneck-latent-blocks 3
--bottleneck-mixer-dim 512
--decoder-dim 512
--decoder-blocks 4
--n-c 32
--lr-bottleneck 1e-4
--lr-coarse-flow 1e-4
--lr-decoder 1e-4
--log-every 50
--diag-every 500
```

Resource paths:

```text
--checkpoint-dir /workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var_preflight
--provenance-out /workspace/preflight/inv016_unwhitened_siglip2_m512_cov_var/ego4d_siglip2_m512_cov_var_resource.json
```

## 4. Exact paid launch additions

Add these to the shared arguments and run in foreground inside detached tmux with
`CUDA_VISIBLE_DEVICES=0`, `PYTHONUNBUFFERED=1`, and `set -o pipefail`:

```text
--wandb-entity smahalanobis-uc-davis
--wandb-project hjepa-vwm
--wandb-group inv016_unwhitened_m512_geometry
--wandb-name "Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 covariance plus variance"
--checkpoint-dir /workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var
--provenance-out /workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var/run_provenance.json
--require-wandb
```

Do not launch on a collision. After launch, verify exact PID/config, GPU ownership, W&B ID/name/
group, first training row, and first diagnostic row. Completion requires exit zero, W&B `finished`,
the step-15,000 checkpoint and its SHA-256, resolved provenance, and complete expected history.
