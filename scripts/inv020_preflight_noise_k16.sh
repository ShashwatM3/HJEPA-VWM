#!/usr/bin/env bash
set -euo pipefail

FLOW_BOTTLENECK_CHECKPOINT=/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt

python train.py \
  --resource-preflight \
  --provenance-out logs/preflight/inv020_fixed_run60_noise_k16.json \
  --data ego4d \
  --encoder dinov3_vitb16 \
  --encoder-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --encoder-precision bf16 \
  --encoder-frame-microbatch 32 \
  --seed 42 \
  --steps 15000 \
  --batch-size 64 \
  --horizon-k 16 \
  --flow-source noise \
  --flow-bottleneck-checkpoint "${FLOW_BOTTLENECK_CHECKPOINT}" \
  --n-c 64 \
  --d-c 512 \
  --bottleneck-mixer-dim 512 \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --condition-dropout 0.0 \
  --lambda-var 0 \
  --lambda-cov 0 \
  --lambda-recon 0 \
  --lambda-recon-pred 0 \
  --lr-coarse-flow 1e-4 \
  --checkpoint-dir /workspace/ckpt/inv020_fixed_run60_noise_k16 \
  --log-every 50 \
  --diag-every 500 \
  --wandb-entity smahalanobis-uc-davis \
  --wandb-project hjepa-vwm \
  --wandb-group inv020_fixed_run60_source_comparison \
  --wandb-name "Investigation 20 · Fixed Run 60 flow source · k=16 Gaussian control"
