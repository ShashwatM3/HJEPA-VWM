#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/workspace/hierarchal-jepa-flow-world-model
GROUP=inv016_unwhitened_internal_memory_width
PREFLIGHT_ROOT=/workspace/preflight/inv016_internal_memory_width
VJEPA_REV=b3c1679b7c34d3255ef3547f27c7b226aefab26f

cd "$ROOT"
mkdir -p /workspace/hf_cache /workspace/ckpt "$PREFLIGHT_ROOT" logs
export HF_HOME=/workspace/hf_cache
export PYTHONHASHSEED=42

if test -n "$(git status --porcelain --untracked-files=no)"; then
  echo "STOP: the remote repository has tracked changes"
  git status --short --untracked-files=no
  exit 2
fi

if pgrep -af '[p]ython.*train.py' >/dev/null; then
  echo "STOP: another train.py process is already running"
  pgrep -af '[p]ython.*train.py'
  exit 3
fi

for width in 512 1024; do
  checkpoint_dir="/workspace/ckpt/inv016_unwhitened_memory_m${width}"
  if test -d "$checkpoint_dir" && find "$checkpoint_dir" -maxdepth 1 -type f -print -quit | grep -q .; then
    echo "STOP: output directory already contains files: $checkpoint_dir"
    exit 4
  fi
done

wandb login --verify
nvidia-smi -L
test -f /workspace/data/ego4d/chunk_manifest.json
test -d /workspace/data/ego4d/train
test -d /workspace/data/ego4d/validation

COMMON_ARGS=(
  --data ego4d
  --steps 15000
  --seed 42
  --encoder vjepa2_vitl16
  --encoder-revision "$VJEPA_REV"
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
  --lambda-var 0
  --lambda-cov 0
  --lambda-sigreg 0
  --lambda-slot 0
  --bottleneck-latent-blocks 3
  --decoder-dim 512
  --decoder-blocks 4
  --n-c 32
  --lr-bottleneck 1e-4
  --lr-coarse-flow 1e-4
  --lr-decoder 1e-4
  --log-every 50
  --diag-every 500
)

echo "STAGE0_START width=1024"
PYTHONUNBUFFERED=1 python train.py \
  --stage0-only \
  "${COMMON_ARGS[@]}" \
  --bottleneck-mixer-dim 1024 \
  2>&1 | tee logs/inv016_internal_memory_stage0_m1024.log
echo "STAGE0_DONE width=1024"

echo "RESOURCE_PREFLIGHT_START width=1024 batch=64"
PYTHONUNBUFFERED=1 python train.py \
  --resource-preflight \
  "${COMMON_ARGS[@]}" \
  --bottleneck-mixer-dim 1024 \
  --checkpoint-dir /workspace/ckpt/inv016_internal_memory_preflight_m1024 \
  --provenance-out "$PREFLIGHT_ROOT/ego4d_m1024_resource.json" \
  2>&1 | tee logs/inv016_internal_memory_resource_m1024.log
echo "RESOURCE_PREFLIGHT_DONE width=1024 batch=64"

run_arm() {
  local width="$1"
  local tag="inv016_unwhitened_memory_m${width}"
  local checkpoint_dir="/workspace/ckpt/${tag}"
  local log="logs/${tag}.log"
  local display_name="Investigation 16 · Internal memory width · EGO4D M=${width}"

  mkdir -p "$checkpoint_dir"
  echo "ARM_START width=$width tag=$tag"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
    "${COMMON_ARGS[@]}" \
    --bottleneck-mixer-dim "$width" \
    --wandb-entity smahalanobis-uc-davis \
    --wandb-project hjepa-vwm \
    --wandb-group "$GROUP" \
    --wandb-name "$display_name" \
    --checkpoint-dir "$checkpoint_dir" \
    --provenance-out "$checkpoint_dir/run_provenance.json" \
    --require-wandb \
    2>&1 | tee "$log"

  test -f "$checkpoint_dir/phase1_step15000.pt"
  test -f "$checkpoint_dir/run_provenance.json"
  sha256sum "$checkpoint_dir/phase1_step15000.pt"
  echo "ARM_DONE width=$width tag=$tag"
}

run_arm 512
run_arm 1024

echo "QUEUE_DONE group=$GROUP"
