# Guide — Investigation 20 paired full-prediction launch

This is the exact two-GPU launch path. It keeps only gates that can prevent an invalid, incompatible,
or immediately failing paid pair. Local full tests must pass before the launch commit is pushed;
the pod reruns only the narrow transfer/config contracts.

Required CLI:

```text
--warm-start-from <present-only checkpoint>
--temporal-target {residual,full_latent}
```

`--resume` is not a substitute: it restores continuation state that this transfer must keep fresh.

## Fixed identities

```bash
export INV020_SOURCE_CHECKPOINT=/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt
export INV020_SOURCE_SHA256=931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1
export INV020_SOURCE_WANDB_ID=60yaqw6d
export INV020_SOURCE_FEATURE_FINGERPRINT=963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415
export INV020_SOURCE_DATASET_FINGERPRINT=df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c
export INV020_ENCODER=dinov3_vitb16
export INV020_N_C=64
export INV020_D_C=512
export INV020_M=512
export INV020_OUTPUT=/workspace/ckpt/inv020_pretrained_bottleneck_temporal_target
export HF_HOME=/workspace/hf_cache
```

Both arms must use these exact values.

## 1. Synchronize the launch commit

The fresh-pod bootstrap may have skipped repository setup. Clone only when absent; otherwise preserve
and inspect the existing checkout.

```bash
if [ ! -d /workspace/hierarchal-jepa-flow-world-model/.git ]; then
  git clone https://github.com/ShashwatM3/HJEPA-VWM.git /workspace/hierarchal-jepa-flow-world-model
fi
cd /workspace/hierarchal-jepa-flow-world-model
git status --short --branch
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull --ff-only origin phase1-v0.2-frozen-encoder
git rev-parse HEAD
git status --short --branch
```

Stop on overlapping remote edits. Never reset or clean them.

## 2. Essential environment, source, and recipe gate

```bash
cd /workspace/hierarchal-jepa-flow-world-model
test -d /workspace/data/ego4d/train
test -d /workspace/data/ego4d/validation
test -f /workspace/data/ego4d/chunk_manifest.json
test -f /workspace/ego4d_raw/manifests/selection_manifest.json
test -s "$INV020_SOURCE_CHECKPOINT"
test "$(nvidia-smi -L | wc -l)" -ge 2
python3 -c "import torch, transformers, decord, wandb; assert transformers.__version__ == '4.57.6'; print('DEPENDENCIES_OK')"
python3 train.py --help | grep -E -- '--warm-start-from|--temporal-target'
pytest -q tests/test_encoder_run_contract.py tests/test_experiment_config.py tests/test_provenance.py
```

One fast Python gate verifies the registered checkpoint, its transfer identity, and the active
full-prediction YAML before either GPU does model work:

```bash
python3 - <<'PY'
import os
from pathlib import Path

import torch
from config import load_experiment_config
from provenance import sha256_file
from train import EXPERIMENT_CONFIG_PATH, finalize_training_config

path = Path(os.environ["INV020_SOURCE_CHECKPOINT"])
digest = sha256_file(path)
assert digest == os.environ["INV020_SOURCE_SHA256"], digest
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["schema"] == "hjepa-phase1-checkpoint-v2"
assert checkpoint["next_step"] == 15000
source = checkpoint["config"]
assert source["encoder"]["alias"] == os.environ["INV020_ENCODER"]
assert source["model"]["n_c"] == int(os.environ["INV020_N_C"])
assert source["model"]["d_c"] == int(os.environ["INV020_D_C"])
assert source["model"]["bottleneck_mixer_dim"] == int(os.environ["INV020_M"])
assert source["model"]["bottleneck_latent_blocks"] == 3
assert source["model"]["decoder_dim"] == 512
assert source["model"]["decoder_blocks"] == 4
assert source["train"]["present_recon_only"] is True
assert source["train"]["whiten_features"] is False
assert source["train"]["recon_residual_target"] is False
assert checkpoint["wandb_run_id"] == os.environ["INV020_SOURCE_WANDB_ID"]
assert checkpoint["feature_fingerprint"] == os.environ["INV020_SOURCE_FEATURE_FINGERPRINT"]
assert checkpoint["dataset_identity"]["fingerprint"] == os.environ["INV020_SOURCE_DATASET_FINGERPRINT"]
assert "bottleneck" in checkpoint and "decoder" in checkpoint

cfg = load_experiment_config(EXPERIMENT_CONFIG_PATH).config
finalize_training_config(cfg)
assert not cfg.train.present_recon_only
assert cfg.train.lambda_recon == 1.0 and cfg.train.lambda_recon_pred == 0.0
assert cfg.train.recon_loss_mode == "cosine" and not cfg.train.recon_residual_target
assert cfg.train.lambda_var == 0.5 and cfg.train.lambda_cov == 0.01
assert cfg.train.lambda_sigreg == 0.0 and cfg.train.lambda_slot == 0.0
assert not cfg.train.whiten_features
assert cfg.train.horizon_k == 12
assert cfg.train.global_batch == 64
assert cfg.train.max_steps == cfg.train.stage1_steps == 15000
assert cfg.model.bottleneck_latent_blocks == 3
assert cfg.model.decoder_dim == 512 and cfg.model.decoder_blocks == 4
print("INV020_SOURCE_AND_RECIPE_OK", digest)
PY
```

## 3. Run both one-step resource gates concurrently

These are the only pre-launch GPU jobs. They exercise the exact warm start, full batch, future
branch, backward pass, optimizer step, and memory boundary. Running them concurrently avoids a
serial two-arm delay.

The full EGO4D identity opens every clip to bind its frame count. The implementation performs those
metadata reads with 16 bounded workers while preserving sorted canonical output. Expect several
minutes with empty logs and idle GPUs before model allocation; active CPU/I/O is normal and is not
a reason to terminate the gate.

```bash
mkdir -p logs/inv020_preflight
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py \
  --data ego4d --encoder "$INV020_ENCODER" \
  --n-c "$INV020_N_C" --d-c "$INV020_D_C" --bottleneck-mixer-dim "$INV020_M" \
  --warm-start-from "$INV020_SOURCE_CHECKPOINT" --temporal-target residual \
  --resource-preflight --provenance-out logs/inv020_preflight/residual.json \
  > logs/inv020_preflight/residual.log 2>&1 &
INV020_PREFLIGHT_RESIDUAL_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 train.py \
  --data ego4d --encoder "$INV020_ENCODER" \
  --n-c "$INV020_N_C" --d-c "$INV020_D_C" --bottleneck-mixer-dim "$INV020_M" \
  --warm-start-from "$INV020_SOURCE_CHECKPOINT" --temporal-target full_latent \
  --resource-preflight --provenance-out logs/inv020_preflight/full_latent.json \
  > logs/inv020_preflight/full_latent.log 2>&1 &
INV020_PREFLIGHT_FULL_PID=$!

wait "$INV020_PREFLIGHT_RESIDUAL_PID"
wait "$INV020_PREFLIGHT_FULL_PID"
python3 train.py --compare-temporal-target-provenance \
  logs/inv020_preflight/residual.json logs/inv020_preflight/full_latent.json
tail -n 20 logs/inv020_preflight/residual.log
tail -n 20 logs/inv020_preflight/full_latent.log
```

Both jobs must exit zero and the parity comparator must prove that only
`train.predict_residual` differs. If either OOMs, do not launch or lower only one arm.

## 4. Collision gate and parallel launch

```bash
test ! -e "$INV020_OUTPUT"
test ! -e logs/inv020_residual.log
test ! -e logs/inv020_full_latent.log
tmux has-session -t inv020_residual 2>/dev/null && exit 1 || true
tmux has-session -t inv020_full_latent 2>/dev/null && exit 1 || true
mkdir -p "$INV020_OUTPUT/residual" "$INV020_OUTPUT/full_latent" logs
```

Residual target on GPU 0:

```bash
tmux new-session -d -s inv020_residual "cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && export HF_HOME=/workspace/hf_cache && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder dinov3_vitb16 --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 --warm-start-from /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt --temporal-target residual --checkpoint-dir /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/residual --provenance-out /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/residual/run_provenance.json --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm --wandb-group inv020_pretrained_bottleneck_temporal_target --wandb-name 'Investigation 20 · Temporal target · Residual prediction, DINOv3 64 slots by 512' --require-wandb 2>&1 | tee /workspace/hierarchal-jepa-flow-world-model/logs/inv020_residual.log"
```

Full-latent target on GPU 1:

```bash
tmux new-session -d -s inv020_full_latent "cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && export HF_HOME=/workspace/hf_cache && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder dinov3_vitb16 --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 --warm-start-from /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt --temporal-target full_latent --checkpoint-dir /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/full_latent --provenance-out /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/full_latent/run_provenance.json --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm --wandb-group inv020_pretrained_bottleneck_temporal_target --wandb-name 'Investigation 20 · Temporal target · Full-latent prediction, DINOv3 64 slots by 512' --require-wandb 2>&1 | tee /workspace/hierarchal-jepa-flow-world-model/logs/inv020_full_latent.log"
```

## 5. Immediate launch proof

```bash
tmux list-sessions
pgrep -af "python.*train.py"
nvidia-smi
tail -n 120 logs/inv020_residual.log
tail -n 120 logs/inv020_full_latent.log
python3 train.py --compare-temporal-target-provenance \
  "$INV020_OUTPUT/residual/run_provenance.json" \
  "$INV020_OUTPUT/full_latent/run_provenance.json"
```

Require both processes, one occupied GPU each, fresh W&B IDs/URLs, the same source checkpoint
SHA-256, global step 0, `prediction_active=1`, nonzero finite `L_flow`, no skipped update, residual
`predict_residual=true`, and full-latent `predict_residual=false`. Record the two W&B IDs in the run
descriptions only after this evidence exists.

## 6. Recovery boundary

- Reconnect and inspect tmux after SSH loss; do not relaunch blindly.
- Resume an interrupted arm only from its own Investigation-020 checkpoint and W&B ID.
- Never resume either arm from the Investigation-019 source.
- A source hash, provenance-parity, scientific-config, nonfinite, or corruption failure invalidates
  the pair. An OOM requires one registered common batch change for both arms.
- Do not kill unrelated processes or alter a live scientific recipe.
