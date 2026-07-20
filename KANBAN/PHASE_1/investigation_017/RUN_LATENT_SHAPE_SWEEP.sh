#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/workspace/hierarchal-jepa-flow-world-model
GROUP=inv017_raw_latent_shape_cov_var
PREFLIGHT_ROOT=/workspace/preflight/inv017_raw_latent_shape
VJEPA_REV=b3c1679b7c34d3255ef3547f27c7b226aefab26f
EXPECTED_COMMIT=${INV017_EXPECTED_COMMIT:?set INV017_EXPECTED_COMMIT to the published tested SHA}

cd "$ROOT"
mkdir -p /workspace/hf_cache /workspace/ckpt "$PREFLIGHT_ROOT" logs
export HF_HOME=/workspace/hf_cache
export PYTHONHASHSEED=42

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
python train.py --help | grep -q -- '--d-c'

if pgrep -af '[p]ython.*train.py' >/dev/null; then
  echo "STOP: another train.py process is already running"
  pgrep -af '[p]ython.*train.py'
  exit 3
fi

ARMS=(
  "069 32 256"
  "070 16 512"
  "071 64 128"
  "072 16 128"
  "073 16 256"
  "074 32 128"
  "075 32 512"
  "076 64 256"
  "077 64 512"
)

for arm in "${ARMS[@]}"; do
  read -r run_number n_c d_c <<< "$arm"
  tag="run${run_number}_inv017_raw_shape_n${n_c}_d${d_c}_s42"
  checkpoint_dir="/workspace/ckpt/${tag}"
  if test -d "$checkpoint_dir" && find "$checkpoint_dir" -maxdepth 1 -type f -print -quit | grep -q .; then
    echo "STOP: output directory already contains files: $checkpoint_dir"
    exit 4
  fi
  if test -e "logs/${tag}.log"; then
    echo "STOP: output log already exists: logs/${tag}.log"
    exit 4
  fi
done

test ! -e logs/inv017_raw_shape_stage0_n64_d512.log
test ! -e logs/inv017_raw_shape_resource_n64_d512.log
if test -d /workspace/ckpt/inv017_raw_shape_preflight_n64_d512; then
  test -z "$(find /workspace/ckpt/inv017_raw_shape_preflight_n64_d512 -maxdepth 1 -type f -print -quit)"
fi
test ! -e "$PREFLIGHT_ROOT/ego4d_n64_d512_resource.json"

wandb login --verify
nvidia-smi -L
test -f /workspace/data/ego4d/chunk_manifest.json
test -d /workspace/data/ego4d/train
test -d /workspace/data/ego4d/validation

COMMON_ARGS=(
  --data ego4d --steps 15000 --seed 42
  --encoder vjepa2_vitl16 --encoder-revision "$VJEPA_REV"
  --encoder-precision bf16 --encoder-frame-microbatch 8
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache
  --batch-size 64 --horizon-k 12 --present-recon-only
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01
  --lambda-sigreg 0 --lambda-slot 0
  --bottleneck-latent-blocks 3 --bottleneck-mixer-dim 512
  --decoder-dim 512 --decoder-blocks 4
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4
  --log-every 50 --diag-every 500
)

PYTHONUNBUFFERED=1 python train.py \
  --stage0-only "${COMMON_ARGS[@]}" --n-c 64 --d-c 512 \
  2>&1 | tee logs/inv017_raw_shape_stage0_n64_d512.log

PYTHONUNBUFFERED=1 python train.py \
  --resource-preflight "${COMMON_ARGS[@]}" --n-c 64 --d-c 512 \
  --checkpoint-dir /workspace/ckpt/inv017_raw_shape_preflight_n64_d512 \
  --provenance-out "$PREFLIGHT_ROOT/ego4d_n64_d512_resource.json" \
  2>&1 | tee logs/inv017_raw_shape_resource_n64_d512.log

run_arm() {
  local run_number="$1"
  local n_c="$2"
  local d_c="$3"
  local tag="run${run_number}_inv017_raw_shape_n${n_c}_d${d_c}_s42"
  local checkpoint_dir="/workspace/ckpt/${tag}"
  local display_name="Run ${run_number#0} · Investigation 17 · Raw latent shape · N=${n_c} D=${d_c}"

  mkdir -p "$checkpoint_dir"
  echo "ARM_START n_c=$n_c d_c=$d_c tag=$tag"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
    "${COMMON_ARGS[@]}" --n-c "$n_c" --d-c "$d_c" \
    --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
    --wandb-group "$GROUP" --wandb-name "$display_name" \
    --checkpoint-dir "$checkpoint_dir" \
    --provenance-out "$checkpoint_dir/run_provenance.json" --require-wandb \
    2>&1 | tee "logs/${tag}.log"

  test -f "$checkpoint_dir/phase1_step15000.pt"
  test -f "$checkpoint_dir/run_provenance.json"
  sha256sum "$checkpoint_dir/phase1_step15000.pt"
  echo "ARM_DONE n_c=$n_c d_c=$d_c tag=$tag"
}

for arm in "${ARMS[@]}"; do
  read -r run_number n_c d_c <<< "$arm"
  run_arm "$run_number" "$n_c" "$d_c"
done

echo "QUEUE_DONE group=$GROUP"
