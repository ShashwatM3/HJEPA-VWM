#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/workspace/hierarchal-jepa-flow-world-model
CHECKPOINT_ROOT=/workspace/ckpt/inv019_bottleneck_shape
WANDB_ENTITY=smahalanobis-uc-davis
WANDB_PROJECT=hjepa-vwm
WANDB_GROUP=inv019_three_encoder_bottleneck_shape
EXPECTED_SHA=${INV019_EXPECTED_SHA:?INV019_EXPECTED_SHA must be set to the synchronized commit SHA}

ARM_NAMES=(tight width_heavy slot_heavy expanded)
ARM_N=(16 16 64 64)
ARM_D=(128 512 128 512)
ARM_M=512

cd "$REPO"
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1

actual_sha=$(git rev-parse HEAD)
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  printf 'SOURCE_MISMATCH expected=%s actual=%s\n' "$EXPECTED_SHA" "$actual_sha"
  exit 2
fi
git diff --quiet
git diff --cached --quiet
if (( $(nvidia-smi -L | wc -l) < 4 )); then
  printf 'GPU_COUNT_FAIL expected_at_least=4\n'
  exit 2
fi
if [[ -e "$CHECKPOINT_ROOT" ]]; then
  printf 'CHECKPOINT_COLLISION path=%s\n' "$CHECKPOINT_ROOT"
  exit 2
fi
mkdir -p "$CHECKPOINT_ROOT" logs

run_arm() {
  local encoder=$1
  local encoder_label=$2
  local lane_slug=$3
  local arm_index=$4
  local gpu=$arm_index
  local arm=${ARM_NAMES[$arm_index]}
  local n_c=${ARM_N[$arm_index]}
  local d_c=${ARM_D[$arm_index]}
  local run_slug="inv019_${lane_slug}_n${n_c}_d${d_c}_m${ARM_M}"
  local checkpoint_dir="$CHECKPOINT_ROOT/$run_slug"
  local provenance_out="$checkpoint_dir/run_provenance.json"
  local log_path="logs/${run_slug}.log"
  local wandb_name="Investigation 19 · Bottleneck capacity · ${encoder_label} ${n_c} slots, slot width ${d_c}, memory ${ARM_M}"

  if [[ -e "$checkpoint_dir" || -e "$log_path" ]]; then
    printf 'ARM_COLLISION encoder=%s arm=%s checkpoint=%s log=%s\n' \
      "$encoder" "$arm" "$checkpoint_dir" "$log_path"
    return 2
  fi
  mkdir -p "$checkpoint_dir"

  printf 'ARM_START encoder=%s arm=%s gpu=%s N=%s D=%s M=%s checkpoint=%s\n' \
    "$encoder" "$arm" "$gpu" "$n_c" "$d_c" "$ARM_M" "$checkpoint_dir"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python3 train.py \
    --data ego4d \
    --encoder "$encoder" \
    --n-c "$n_c" \
    --d-c "$d_c" \
    --bottleneck-mixer-dim "$ARM_M" \
    --checkpoint-dir "$checkpoint_dir" \
    --provenance-out "$provenance_out" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-name "$wandb_name" \
    --require-wandb \
    2>&1 | tee "$log_path"
  local train_status=${PIPESTATUS[0]}
  set -e

  if (( train_status != 0 )); then
    printf 'ARM_FAIL encoder=%s arm=%s status=%s log=%s\n' \
      "$encoder" "$arm" "$train_status" "$log_path"
    return "$train_status"
  fi

  local final_checkpoint="$checkpoint_dir/phase1_step15000.pt"
  if [[ ! -s "$final_checkpoint" || ! -s "$provenance_out" ]]; then
    printf 'ARM_EVIDENCE_FAIL encoder=%s arm=%s checkpoint=%s provenance=%s\n' \
      "$encoder" "$arm" "$final_checkpoint" "$provenance_out"
    return 3
  fi

  local checksum
  checksum=$(sha256sum "$final_checkpoint" | awk '{print $1}')
  printf 'ARM_DONE encoder=%s arm=%s gpu=%s checkpoint=%s sha256=%s\n' \
    "$encoder" "$arm" "$gpu" "$final_checkpoint" "$checksum"
}

run_lane() {
  local encoder=$1
  local encoder_label=$2
  local lane_slug=$3
  local pids=()
  local arm_index
  local failed=0

  printf 'LANE_START encoder=%s label=%s sha=%s\n' "$encoder" "$encoder_label" "$actual_sha"
  for arm_index in "${!ARM_NAMES[@]}"; do
    run_arm "$encoder" "$encoder_label" "$lane_slug" "$arm_index" &
    pids+=("$!")
  done

  for arm_index in "${!pids[@]}"; do
    if wait "${pids[$arm_index]}"; then
      :
    else
      status=$?
      failed=1
      printf 'LANE_ARM_NONZERO encoder=%s arm=%s status=%s\n' \
        "$encoder" "${ARM_NAMES[$arm_index]}" "$status"
    fi
  done

  if (( failed != 0 )); then
    printf 'LANE_FAIL encoder=%s\n' "$encoder"
    return 1
  fi
  printf 'LANE_DONE encoder=%s\n' "$encoder"
}

printf 'SWEEP_START sha=%s group=%s\n' "$actual_sha" "$WANDB_GROUP"
run_lane vjepa2_vitl16 V-JEPA2 vjepa2
run_lane dinov3_vitb16 DINOv3 dinov3
run_lane siglip2_vitb16 'SigLIP 2' siglip2
printf 'SWEEP_DONE sha=%s group=%s\n' "$actual_sha" "$WANDB_GROUP"
