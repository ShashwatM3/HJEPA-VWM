#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/workspace/hierarchal-jepa-flow-world-model
EXPECTED_COMMIT=${INV017_EXPECTED_COMMIT:?set INV017_EXPECTED_COMMIT to the published tested SHA}
LANE=${1:?usage: RUN_LATENT_SHAPE_SWEEP.sh vjepa2|siglip2|dinov3}

case "$LANE" in
  vjepa2)
    ENCODER_ALIAS=vjepa2_vitl16
    ENCODER_LABEL=V-JEPA2
    RECIPE=configs/inv017_latent_shape.yaml
    ;;
  siglip2)
    ENCODER_ALIAS=siglip2_vitb16
    ENCODER_LABEL="SigLIP 2"
    RECIPE=configs/inv017_latent_shape.yaml
    ;;
  dinov3)
    ENCODER_ALIAS=dinov3_vitb16
    ENCODER_LABEL=DINOv3
    RECIPE=configs/inv017_dinov3_latent_shape.yaml
    ;;
  *)
    echo "usage: RUN_LATENT_SHAPE_SWEEP.sh vjepa2|siglip2|dinov3" >&2
    exit 2
    ;;
esac

GROUP="inv017_${LANE}_latent_shape_cov_var"

cd "$ROOT"
mkdir -p /workspace/hf_cache /workspace/ckpt logs
export HF_HOME=/workspace/hf_cache
export PYTHONHASHSEED=42

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
python train.py --help | grep -q -- '--config'
test -f "$RECIPE"

if pgrep -af '[p]ython.*train.py' >/dev/null; then
  echo "STOP: another train.py process is already running"
  pgrep -af '[p]ython.*train.py'
  exit 3
fi

case "$LANE" in
  vjepa2|siglip2)
    # The exact geometry-active 32x256 centers already exist as guiduvjp (V-JEPA2)
    # and ufbeokj2 (SigLIP 2). Reuse those observations instead of paying for
    # duplicate configurations.
    ARMS=(
      "16 512"
      "64 128"
      "16 128"
      "16 256"
      "32 128"
      "32 512"
      "64 256"
      "64 512"
    )
    ;;
  dinov3)
    # Run 69 had variance/covariance disabled, so every geometry-active DINOv3
    # shape, including the center, is new evidence.
    ARMS=(
      "32 256"
      "16 512"
      "64 128"
      "16 128"
      "16 256"
      "32 128"
      "32 512"
      "64 256"
      "64 512"
    )
    ;;
esac

for arm in "${ARMS[@]}"; do
  read -r n_c d_c <<< "$arm"
  tag="inv017_${LANE}_raw_shape_n${n_c}_d${d_c}_s42"
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

wandb login --verify
nvidia-smi -L
test -f /workspace/data/ego4d/chunk_manifest.json
test -d /workspace/data/ego4d/train
test -d /workspace/data/ego4d/validation

COMMON_ARGS=(
  --config "$RECIPE"
  --data ego4d
  --encoder "$ENCODER_ALIAS"
  --bottleneck-mixer-dim 512
)

run_arm() {
  local n_c="$1"
  local d_c="$2"
  local tag="inv017_${LANE}_raw_shape_n${n_c}_d${d_c}_s42"
  local checkpoint_dir="/workspace/ckpt/${tag}"
  local display_name="Investigation 17 · ${ENCODER_LABEL} latent shape · N=${n_c}, D=${d_c}"

  mkdir -p "$checkpoint_dir"
  echo "ARM_START encoder=$ENCODER_ALIAS n_c=$n_c d_c=$d_c tag=$tag"
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
  echo "ARM_DONE encoder=$ENCODER_ALIAS n_c=$n_c d_c=$d_c tag=$tag"
}

for arm in "${ARMS[@]}"; do
  read -r n_c d_c <<< "$arm"
  run_arm "$n_c" "$d_c"
done

echo "QUEUE_DONE encoder=$ENCODER_ALIAS group=$GROUP"
