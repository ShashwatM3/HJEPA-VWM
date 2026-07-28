# GUIDE — Run 68, DINOv3 whitened internal-memory M=512 covariance plus variance (EGO4D)

> **Local-label/status correction (2026-07-27):** this historical launch guide corresponds to
> canonical local [Run 070](../run_070_whitened_dinov3_m512_cov_var/), W&B `qqozribu`.
> Use that folder for completed evidence; do not relaunch this retained recipe.

This is the exact gated launch guide for the whitened counterpart to completed DINOv3 Run 67
([`../run_067_dinov3_unwhitened_internal_memory_m512/`](../run_067_dinov3_unwhitened_internal_memory_m512/),
W&B [`it7sq8nz`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/it7sq8nz)). Preserve the
entire Run 67 recipe and change only the registered whitening bundle. Run every section in order;
stop at the first failed gate. Do not use `--resume`.

## 0. Values used by every section

Paste this block at the start of the SSH shell and again inside every new tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model

export DINO_REV=5931719e67bbdb9737e363e781fb0c67687896bc
export EXPECTED_GIT_SHA=083cf8a6e87168702efe46ac6bfe485756dcb439
export BATCH=64
export FRAME_MB=32
export STATS=/workspace/stats/inv016_encoder_substrate/dinov3_vitb16_ego4d_train_seed42.pt
export PREFLIGHT_DIR=/workspace/preflight/inv016_dinov3_whitened_memory_m512_cov_var
export CKPT_DIR=/workspace/ckpt/inv016_dinov3_whitened_memory_m512_cov_var
export SMOKE_CKPT_DIR=/workspace/ckpt/inv016_dinov3_whitened_memory_m512_cov_var_smoke
export LOG=logs/inv016_dinov3_whitened_memory_m512_cov_var.log
export PYTHONHASHSEED=42
export HF_HOME=/workspace/hf_cache

mkdir -p /workspace/hf_cache "$(dirname "$STATS")" "$PREFLIGHT_DIR" \
  "$CKPT_DIR" "$SMOKE_CKPT_DIR" logs
```

`FRAME_MB=32` is the completed Run 67 value and the first candidate. If resource preflight requires
`16`, `8`, `4`, `2`, or `1`, use the largest value that passes with headroom and keep it identical
for adapter smoke, whitening-stat generation, artifact validation, every preflight, smoke, and final
training. Do not lower physical `BATCH=64`.

## 1. Environment and Git-state verification

Do not discard unexplained pod changes.

```bash
git status --short --branch
test -z "$(git status --porcelain=v1)"
git fetch origin
git switch codex/task3-dino-run066
git pull --ff-only
test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_SHA"

source .venv/bin/activate
python --version
which python
python -m pip --version
python -m pip install -r requirements.txt
python - <<'PY'
import transformers
assert transformers.__version__ == "4.57.6", transformers.__version__
print("transformers", transformers.__version__)
PY
python -m pytest -q
python -m black --check encoders.py train.py tests/test_encoders.py
python -m ruff check encoders.py train.py tests/test_encoders.py
wandb login --verify
nvidia-smi
```

Verify the worktree is clean, HEAD is exactly the expected commit, Python comes from the existing
repository `.venv`, all tests/static checks pass, W&B verifies, and the intended GPU is idle.

## 2. Confirm the completed unwhitened DINOv3 baseline

```bash
test -f /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt
test -f /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/run_provenance.json
test "$(sha256sum /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt | awk '{print $1}')" = \
  f4513c9ce01e18f4db6cc6c3434c5a5f53749019f36992ed332fc14e67dd33db
```

W&B baseline identity must remain Run 67 / `it7sq8nz`, state `finished`, display name
`Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512`. This checkpoint is a
comparison artifact only. **Never pass it to `--resume`.**

## 3. Prove the real DINOv3 adapter on CUDA

```bash
python encoders.py --smoke \
  --encoder dinov3_vitb16 \
  --revision "$DINO_REV" \
  --precision bf16 \
  --frame-microbatch "$FRAME_MB" \
  --attention-implementation sdpa \
  --hf-cache-dir /workspace/hf_cache \
  --batch-size 1 \
  --device cuda \
  | tee "$PREFLIGHT_DIR/dino_adapter_smoke.json"
```

Verify requested and resolved revisions both equal `$DINO_REV`, trainable encoder parameters are
zero, output is finite with shape `[1,2048,768]`, device is CUDA, precision is bf16, and the feature
layout is `8x16x16` in `time_y_x` order.

## 4. Revalidate full EGO4D completeness

```bash
python - <<'PY'
from config import Config
from provenance import build_dataset_identity

cfg = Config()
cfg.data.dataset = "ego4d"
identity = build_dataset_identity(cfg, require_complete=True)
assert identity.get("completeness") == "stage-4d-verified", identity
print("dataset:", identity["dataset"])
print("train clips:", identity["splits"]["train"]["count"])
print("validation clips:", identity["splits"]["validation"]["count"])
print("fingerprint:", identity["fingerprint"])
print("completeness:", identity["completeness"])
PY
```

Stop on any manifest, source-UID, split, frame-count, or fingerprint failure. A changed dataset
fingerprint invalidates every earlier whitening artifact.

## 5. Settle frame microbatch before fitting statistics

Run the exact trainable graph without loading a whitening artifact so frame microbatch is settled
before it becomes part of the stats identity:

```bash
python train.py --resource-preflight \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$PREFLIGHT_DIR/unwhitened_ckpt" \
  --provenance-out "$PREFLIGHT_DIR/dino_m512_cov_var_unwhitened_resource.json"
```

Require finite metrics, physical batch 64, positive throughput, and CUDA memory measurements. On
OOM retry only `FRAME_MB=16`, then `8`, `4`, `2`, `1`, rerunning Sections 3 and 5. Stop if batch 64
cannot fit at frame microbatch 1.

## 6. Fit the DINOv3/EGO4D whitening envelope

Do not reuse V-JEPA or SigLIP statistics. If `$STATS` already exists, inspect it first; do not
overwrite an unexplained artifact.

```bash
test ! -e "$STATS" || { echo "STOP: whitening artifact already exists; inspect before reuse"; exit 9; }

tmux has-session -t inv016_run068_stats 2>/dev/null \
  && tmux attach -t inv016_run068_stats \
  || tmux new -s inv016_run068_stats
```

Inside tmux, paste Section 0 again with the final settled `FRAME_MB`, then run:

```bash
python whiten_stats.py \
  --data ego4d --split train \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size 8 --max-clips 12800 --seed 42 --device cuda \
  --output "$STATS"

python whiten_stats.py --inspect "$STATS" | tee "$PREFLIGHT_DIR/dino_whitening_inspect.json"
sha256sum "$STATS" | tee "$PREFLIGHT_DIR/dino_whitening_sha256.txt"
```

Retain the envelope, inspection JSON, payload fingerprint, and SHA-256.

## 7. Strict whitening-artifact metadata validation

```bash
python - "$STATS" "$DINO_REV" "$FRAME_MB" <<'PY'
import json
import sys

from provenance import WHITENING_EIGENSOLVER, load_whitening_envelope

path, revision, frame_mb = sys.argv[1], sys.argv[2], int(sys.argv[3])
envelope = load_whitening_envelope(path)
metadata = envelope["metadata"]
spec = metadata["encoder_spec"]
dataset = metadata["dataset_identity"]

assert spec["family"] == "dinov3", spec
assert spec["requested_revision"] == revision, spec
assert spec["resolved_revision"] == revision, spec
assert spec["feature_dim"] == 768, spec
assert spec["layout"]["temporal"] == 8, spec
assert spec["layout"]["height"] == spec["layout"]["width"] == 16, spec
assert spec["layout"]["order"] == "time_y_x", spec
assert "dinov3" in spec["preprocess_version"], spec
assert metadata["precision"] == "bf16", metadata
assert spec["attention_implementation"] == "sdpa", spec
assert metadata["frame_microbatch"] == frame_mb, metadata
assert spec["frame_microbatch"] == frame_mb, spec
assert metadata["split"] == "train", metadata
assert metadata["transform_seed"] == 42, metadata
assert metadata["clip_count"] == 12800, metadata
assert dataset["dataset"] == "ego4d", dataset
assert dataset.get("completeness") == "stage-4d-verified", dataset
assert metadata["eigensolver"] == WHITENING_EIGENSOLVER, metadata

print(json.dumps({
    "payload_fingerprint": envelope["payload_fingerprint"],
    "encoder": spec["family"],
    "revision": spec["resolved_revision"],
    "preprocess_version": spec["preprocess_version"],
    "dataset": dataset["dataset"],
    "dataset_fingerprint": metadata["dataset_fingerprint"],
    "split": metadata["split"],
    "seed": metadata["transform_seed"],
    "precision": metadata["precision"],
    "attention": spec["attention_implementation"],
    "frame_microbatch": metadata["frame_microbatch"],
    "clip_count": metadata["clip_count"],
}, indent=2, sort_keys=True))
print("DINOv3 whitening metadata contract OK")
PY
```

`whiten_eps` is deliberately not part of the fitted envelope: it is applied when training builds
the whitener. Verify `whiten_eps=1e-4` in Sections 8–10 provenance/config, while the artifact itself
must prove the 12,800-clip fit and all encoder/dataset identities above.

## 8. Whitening-active resource preflight and Stage 0

```bash
python train.py --resource-preflight \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
  --whiten-stats-path "$STATS" \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$PREFLIGHT_DIR/whitened_ckpt" \
  --provenance-out "$PREFLIGHT_DIR/dino_m512_cov_var_whitened_resource.json"

python train.py --stage0-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
  --whiten-stats-path "$STATS" \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$CKPT_DIR"
```

Require `whiten_active=1`, `present_recon_only=1`, prediction inactive, `L_flow=0`,
`L_recon_pred=0`, finite positive `L_var`/`L_cov`, and a successful Stage 0 EMA transition.

## 9. Materialize exact no-step provenance

```bash
python train.py --preflight-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
  --whiten-stats-path "$STATS" \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_whitened_memory \
  --wandb-name "Investigation 16 · Encoder substrate · DINOv3 whitened memory M=512 covariance plus variance" \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$PREFLIGHT_DIR/dino_m512_cov_var_exact.json"

grep -E 'dinov3|5931719e|ego4d|12800|0.0001|Investigation 16' \
  "$PREFLIGHT_DIR/dino_m512_cov_var_exact.json"
```

Compare this JSON with Run 67 provenance. Only the registered whitening bundle and operational
identity may differ; frame microbatch may differ only if Section 5 required the documented ladder.

## 10. Mandatory 100-step full-EGO4D smoke

```bash
test -z "$(find "$SMOKE_CKPT_DIR" -maxdepth 1 \
  \( -name 'phase1_step*.pt' -o -name 'run_provenance.json' \) -print -quit)" \
  || { echo "STOP: smoke checkpoint directory is not empty"; exit 9; }

set -o pipefail
python train.py \
  --data ego4d --steps 100 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
  --whiten-stats-path "$STATS" \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_whitened_memory_smoke \
  --wandb-name "Investigation 16 · Encoder substrate · DINOv3 whitened memory M=512 covariance plus variance 100-step smoke" \
  --checkpoint-dir "$SMOKE_CKPT_DIR" \
  --provenance-out "$SMOKE_CKPT_DIR/run_provenance.json" \
  --require-wandb --log-every 10 --diag-every 50 \
  2>&1 | tee logs/inv016_dinov3_whitened_memory_m512_cov_var_smoke.log

test -f "$SMOKE_CKPT_DIR/phase1_step100.pt"
test -f "$SMOKE_CKPT_DIR/run_provenance.json"
sha256sum "$SMOKE_CKPT_DIR/phase1_step100.pt"
```

Require zero exit, finite metrics, no skipped/NaN update, W&B reaching the final smoke step, and
both files above. Do not resume the paid run from this checkpoint.

## 11. Launch the 15,000-step run

```bash
test -z "$(find "$CKPT_DIR" -maxdepth 1 \
  \( -name 'phase1_step*.pt' -o -name 'run_provenance.json' \) -print -quit)" \
  || { echo "STOP: final checkpoint directory is not empty"; exit 9; }

tmux has-session -t inv016_run068 2>/dev/null && echo "STOP: session exists; inspect it" \
  || tmux new -s inv016_run068
```

Inside tmux, paste Section 0 again with the final `FRAME_MB`, then launch from scratch:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision "$DINO_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
  --whiten-stats-path "$STATS" \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_whitened_memory \
  --wandb-name "Investigation 16 · Encoder substrate · DINOv3 whitened memory M=512 covariance plus variance" \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$CKPT_DIR/run_provenance.json" \
  --require-wandb --log-every 50 --diag-every 500 \
  2>&1 | tee "$LOG"
```

## 12. Early tripwires and monitoring

```bash
tmux list-sessions
pgrep -af '[p]ython.*train.py'
nvidia-smi
tail -n 100 "$LOG"
grep -m1 'step=0 ' "$LOG"
```

At the first logged step require `whiten_active=1`, `present_recon_only=1`, prediction inactive,
residual target off, `M=512`, `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`, SIGReg/slot
weights zero, `L_flow=0`, `L_recon_pred=0`, and `grad_skipped=0`. Confirm W&B config contains the
exact DINO revision, dataset and feature fingerprints, whitening payload fingerprint,
`whiten_eps=1e-4`, expected clips 12,800, physical batch 64, and planned display name.

Ongoing:

```bash
tail -f "$LOG"
watch -n 5 nvidia-smi
tmux capture-pane -p -t inv016_run068 -S -160
```

## 13. Completion proof and W&B verification

After the process exits:

```bash
test -f "$CKPT_DIR/phase1_step15000.pt"
test -f "$CKPT_DIR/run_provenance.json"
sha256sum "$CKPT_DIR/phase1_step15000.pt"
sha256sum "$STATS"
tail -n 20 "$LOG"
pgrep -af '[p]ython.*train.py' || true
nvidia-smi
```

Confirm W&B reached 15,000 training steps, state is `finished`, final summary contains the same
checkpoint path/SHA-256, and both provenance and whitening artifacts are committed. Record the W&B
ID/URL, whitening fingerprint/SHA, final checkpoint SHA, and Reading Cycle B in this run folder.

## 14. Post-run comparison

Compare against Run 67 (`it7sq8nz`) using within-run diagnostics: stability, correct-vs-shuffled
gap, conditioned share, std/cosine/effective-rank/slot-rank trajectories, and late-window medians.
**Do not compare raw whitened and unwhitened loss magnitudes directly:** whitening changes the target
coordinate system. The fixed validation batch remains within-source/exact-chunk.

## 15. Cleanup and pod stop

Do not delete whitening statistics, checkpoints, provenance, or logs. After completion proof and
W&B verification agree, detach from tmux, ensure no training process remains, and stop the pod from
RunPod to release the GPU. `/workspace` is on the persistent network volume and must remain
attached/preserved. Pod termination, volume deletion, checkpoint pruning, or moving artifacts is a
separate human-authorized action and is not part of this guide.
