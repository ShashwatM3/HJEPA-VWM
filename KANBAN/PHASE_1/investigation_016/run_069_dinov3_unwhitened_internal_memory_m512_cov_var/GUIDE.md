# Guide — Run 69, DINOv3 unwhitened internal-memory M=512 covariance plus variance

This retained RunPod recipe completed successfully. Do not relaunch it as part of documentation maintenance.

## Verified completion

- W&B: [`fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6)
- Final checkpoint: `/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/phase1_step15000.pt`
- SHA-256: `fbd4bda4c75780c06b35c837a1291218c4525c1549baf978ef3561eaf81e7cb8`
- Uploaded directories:
  - `s3://hjepa-volume/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/`
    (`7` objects, `2.9 GiB`)
  - `s3://hjepa-volume/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var/`
    (`1` object)

## Retained launch command

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var
mkdir -p /workspace/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var
mkdir -p logs

tmux new-session -d -s inv016_run069 \
  "cd /workspace/hierarchal-jepa-flow-world-model && source .venv/bin/activate && export HF_HOME=/workspace/hf_cache && python train.py \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 \
  --encoder-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --encoder-precision bf16 --encoder-frame-microbatch 32 \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size 64 --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 \
  --recon-loss-mode cosine --recon-warmup-steps 2000 \
  --lambda-var 0.5 --lambda-cov 0.01 --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 --n-c 32 \
  --decoder-dim 512 --decoder-blocks 4 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --log-every 50 --diag-every 500 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_unwhitened_memory \
  --wandb-name 'Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512 covariance plus variance' \
  --checkpoint-dir /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var \
  --provenance-out /workspace/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var/launch_provenance.json \
  --require-wandb 2>&1 | tee /workspace/hierarchal-jepa-flow-world-model/logs/inv016_dinov3_unwhitened_memory_m512_cov_var.log"
```

The command intentionally contains no whitening flag, no whitening stats path, and no `--resume`.
