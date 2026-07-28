# Guide — Investigation 19 three-encoder bottleneck shape sweep

This guide starts paid work quickly from a pod that has completed `NEW_POD.md` through Step 5.
It does not reinstall packages, compute whitening statistics, run Stage 0, or run a full resource
preflight. It performs only fast identity/config/auth/data checks, then launches the queue.

## Fixed identities

- Repository: `/workspace/hierarchal-jepa-flow-world-model`
- Dataset: `/workspace/data/ego4d`
- Hugging Face cache: `/workspace/hf_cache`
- Checkpoints: `/workspace/ckpt/inv019_bottleneck_shape_covvar/`
- Logs: `/workspace/hierarchal-jepa-flow-world-model/logs/inv019_covvar_*.log`
- Controller log: `/workspace/hierarchal-jepa-flow-world-model/logs/inv019_covvar_controller.log`
- tmux: `inv019_bottleneck_shape_covvar`
- W&B entity/project: `smahalanobis-uc-davis/hjepa-vwm`
- W&B group: `inv019_three_encoder_bottleneck_shape_covvar`

## 1. Synchronize one exact clean commit

On the local machine, record the intended source:

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
git status --short --branch
git rev-parse HEAD
git push origin phase1-v0.2-frozen-encoder
```

On the pod, preserve any unexpected work and fast-forward only:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short --branch
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull --ff-only origin phase1-v0.2-frozen-encoder
git rev-parse HEAD
git status --short --branch
```

The local and remote SHAs must match. Stop on overlapping remote changes; never reset or clean them.

## 2. Fast launch gates

These gates should take seconds. They intentionally avoid duplicate setup and corpus-wide preflight
scans.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
test -d /workspace/data/ego4d/train
test -d /workspace/data/ego4d/validation
test -f /workspace/data/ego4d/chunk_manifest.json
test -f /workspace/ego4d_raw/manifests/selection_manifest.json
python3 -c "import torch, transformers, decord, wandb; assert transformers.__version__ == '4.57.6'; print('DEPENDENCIES_OK')"
python3 -c "import wandb; a=wandb.Api(); print('WANDB_OK', a.default_entity)"
nvidia-smi -L
test "$(nvidia-smi -L | wc -l)" -ge 4
```

Verify the resolved scientific recipe without running an encoder:

```bash
python3 - <<'PY'
from config import load_experiment_config
from train import EXPERIMENT_CONFIG_PATH, finalize_training_config

cfg = load_experiment_config(EXPERIMENT_CONFIG_PATH).config
finalize_training_config(cfg)
assert cfg.data.dataset in {"ssv2_tiny", "ego4d"}
assert cfg.train.present_recon_only
assert not cfg.train.whiten_features
assert cfg.train.lambda_recon == 1.0
assert cfg.train.lambda_recon_pred == 0.0
assert cfg.train.lambda_var == 0.5
assert cfg.train.lambda_cov == 0.01
assert cfg.train.lambda_sigreg == 0.0
assert cfg.train.lambda_slot == 0.0
assert cfg.model.bottleneck_mixer_dim == 512
assert cfg.model.bottleneck_latent_blocks == 3
assert cfg.model.decoder_dim == 512 and cfg.model.decoder_blocks == 4
assert cfg.train.max_steps == cfg.train.stage1_steps == 15000
print("INV019_RECIPE_OK")
PY
```

Inspect for collisions before launch:

```bash
tmux list-sessions 2>/dev/null || true
pgrep -af "python.*train.py" || true
test ! -e /workspace/ckpt/inv019_bottleneck_shape_covvar
test ! -e logs/inv019_covvar_controller.log
```

## 3. Launch the controller

Training remains in the foreground inside tmux. The controller starts four V-JEPA2 arms on GPUs
0–3, waits for all four, then does the same for DINOv3 and finally SigLIP 2. It stops before the
next lane if any arm fails.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
export INV019_EXPECTED_SHA="$(git rev-parse HEAD)"
mkdir -p logs
tmux new-session -d -s inv019_bottleneck_shape_covvar "cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && export HF_HOME=/workspace/hf_cache && export INV019_EXPECTED_SHA='$INV019_EXPECTED_SHA' && bash KANBAN/PHASE_1/investigation_019/RUN_SWEEP.sh 2>&1 | tee logs/inv019_covvar_controller.log"
```

To launch only the corrected DINOv3 lane after stopping the mistaken no-geometry lane:

```bash
tmux new-session -d -s inv019_bottleneck_shape_covvar "cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && export HF_HOME=/workspace/hf_cache && export INV019_EXPECTED_SHA='$INV019_EXPECTED_SHA' && export INV019_ONLY_ENCODER=dinov3 && bash KANBAN/PHASE_1/investigation_019/RUN_SWEEP.sh 2>&1 | tee logs/inv019_covvar_controller.log"
```

## 4. Immediate verification

Do not call the launch successful from tmux existence alone.

```bash
tmux list-sessions
pgrep -af "python.*train.py"
nvidia-smi
tail -n 160 logs/inv019_covvar_controller.log
for f in logs/inv019_covvar_*.log; do echo "===== $f"; tail -n 60 "$f"; done
```

Required evidence:

- tmux `inv019_bottleneck_shape_covvar` exists;
- exactly four `train.py` processes for the selected encoder exist;
- GPUs 0–3 each hold one process;
- each log prints its exact shape, resolved encoder revision, raw/unwhitened recipe, and W&B URL;
- for the corrected DINO-only launch, W&B names exactly match:
  - `Investigation 19 · Bottleneck capacity · Covariance + variance · DINOv3 16 slots, slot width 128, memory 512`
  - `Investigation 19 · Bottleneck capacity · Covariance + variance · DINOv3 16 slots, slot width 512, memory 512`
  - `Investigation 19 · Bottleneck capacity · Covariance + variance · DINOv3 64 slots, slot width 128, memory 512`
  - `Investigation 19 · Bottleneck capacity · Covariance + variance · DINOv3 64 slots, slot width 512, memory 512`
- checkpoint and provenance roots are unique per arm;
- the first training row has `present_recon_only=1`, `prediction_active=0`, `whiten_active=0`,
  `L_flow=0`, `L_recon_pred=0`, and `grad_skipped=0`.

The corrected DINO-only controller log must show
`encoder_selection=dinov3` and `LANE_START encoder=dinov3_vitb16`.

## 5. Bounded monitoring

```bash
tmux capture-pane -p -t inv019_bottleneck_shape_covvar -S -200
tail -n 120 logs/inv019_covvar_controller.log
pgrep -af "python.*train.py" || true
nvidia-smi
```

Lane progression markers:

```text
LANE_START encoder=vjepa2_vitl16
LANE_DONE encoder=vjepa2_vitl16
LANE_START encoder=dinov3_vitb16
LANE_DONE encoder=dinov3_vitb16
LANE_START encoder=siglip2_vitb16
LANE_DONE encoder=siglip2_vitb16
SWEEP_DONE
```

Each successful arm must print `ARM_DONE` only after exit 0, its
`phase1_step15000.pt`, provenance sidecar, and checkpoint checksum exist.
An encoder-only recovery launch prints only that encoder's start/done markers.

## 6. Recovery boundaries

- SSH disconnect: reconnect; do not relaunch while tmux exists.
- One arm command/config failure before training: preserve its log, stop only this investigation's
  controller/children if necessary, correct the operational transcription without changing the
  recipe, and relaunch with fresh output identities.
- W&B authentication failure: use the already-approved credential; never print or transfer a key.
- OOM: preserve `nvidia-smi` and the arm log. Do not lower batch, `N_c`, `D_c`, or `M` under the
  same run name.
- External process failure: do not advance the encoder lane. Resume only from a compatible
  same-arm checkpoint if exact resume is verified; otherwise use a new run identity and record why.
- Never kill unrelated sessions or use broad `pkill python`.

## 7. Reconnect command

The RunPod proxy route requires a PTY:

```bash
ssh -tt nuqro8js46cznm-64410eba@ssh.runpod.io -i ~/.ssh/id_ed25519
```

After connecting:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
tmux attach -t inv019_bottleneck_shape_covvar
```
