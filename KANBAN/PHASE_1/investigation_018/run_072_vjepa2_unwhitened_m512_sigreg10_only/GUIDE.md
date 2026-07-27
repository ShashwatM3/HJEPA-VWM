# Guide — run 072 V-JEPA2 unwhitened M=512 with SIGReg only

This guide owns one paid run. Do not change the scientific recipe during execution.

## Identity

```text
W&B entity: smahalanobis-uc-davis
W&B project: hjepa-vwm
W&B group: inv018_vjepa2_sigreg_only
W&B name: Investigation 18 · Regularizer replacement · V-JEPA2 isotropy 10 only
tmux: inv018_vjepa2_sigreg10
log: /workspace/hierarchal-jepa-flow-world-model/logs/inv018_vjepa2_sigreg10.log
checkpoint root: /workspace/ckpt/inv018_vjepa2_unwhitened_m512_sigreg10_only
provenance: /workspace/ckpt/inv018_vjepa2_unwhitened_m512_sigreg10_only/run_provenance.json
```

## 1. Synchronize and inspect

From the local machine, verify the published local SHA, then run the repository's standard
non-interactive SSH preflight. On the pod:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git pull --ff-only origin phase1-v0.2-frozen-encoder
git status --short --branch
git rev-parse HEAD
```

The remote worktree must be clean and its SHA must exactly match the intended local SHA. Inspect
existing processes and sessions before creating anything:

```bash
tmux list-sessions 2>/dev/null || true
pgrep -af "python.*train.py" || true
nvidia-smi
```

Do not overwrite an existing `inv018_vjepa2_sigreg10` session or checkpoint directory.

## 2. Assert the scientific recipe

```bash
python3 - <<'PY'
from config import load_experiment_config

cfg = load_experiment_config("configs/train.yaml").config
expected = {
    "encoder": cfg.encoder.alias,
    "n_c": cfg.model.n_c,
    "d_c": cfg.model.d_c,
    "mixer": cfg.model.bottleneck_mixer_dim,
    "latent_blocks": cfg.model.bottleneck_latent_blocks,
    "decoder_dim": cfg.model.decoder_dim,
    "decoder_blocks": cfg.model.decoder_blocks,
    "present_only": cfg.train.present_recon_only,
    "lambda_recon": cfg.train.lambda_recon,
    "lambda_recon_pred": cfg.train.lambda_recon_pred,
    "lambda_var": cfg.train.lambda_var,
    "lambda_cov": cfg.train.lambda_cov,
    "lambda_sigreg": cfg.train.lambda_sigreg,
    "lambda_slot": cfg.train.lambda_slot,
    "sigreg_warmup": cfg.train.sigreg_warmup_steps,
    "whiten": cfg.train.whiten_features,
}
assert expected == {
    "encoder": "vjepa2_vitl16",
    "n_c": 32,
    "d_c": 256,
    "mixer": 512,
    "latent_blocks": 3,
    "decoder_dim": 512,
    "decoder_blocks": 4,
    "present_only": True,
    "lambda_recon": 1.0,
    "lambda_recon_pred": 0.0,
    "lambda_var": 0.0,
    "lambda_cov": 0.0,
    "lambda_sigreg": 10.0,
    "lambda_slot": 0.0,
    "sigreg_warmup": 2000,
    "whiten": False,
}, expected
print(expected)
PY
```

## 3. Local and remote gates

Run these from the remote repository:

```bash
python3 -m pytest -q
python3 -m py_compile config.py data.py encoders.py models.py losses.py diagnostics.py provenance.py train.py
python3 train.py --stage0-only --data ego4d --encoder vjepa2_vitl16
```

Create the preflight paths:

```bash
mkdir -p /workspace/preflight/inv018_vjepa2_sigreg10 /workspace/ckpt/inv018_vjepa2_sigreg10_resource
```

Materialize exact provenance:

```bash
PYTHONUNBUFFERED=1 python3 train.py \
  --preflight-only \
  --data ego4d \
  --encoder vjepa2_vitl16 \
  --provenance-out /workspace/preflight/inv018_vjepa2_sigreg10/run_provenance.json
```

Run the exact one-step resource preflight:

```bash
PYTHONUNBUFFERED=1 python3 train.py \
  --resource-preflight \
  --data ego4d \
  --encoder vjepa2_vitl16 \
  --checkpoint-dir /workspace/ckpt/inv018_vjepa2_sigreg10_resource \
  --provenance-out /workspace/preflight/inv018_vjepa2_sigreg10/resource_preflight.json
```

Require a successful exit, finite metrics, the pinned V-JEPA2 revision, non-null CUDA peak memory,
`present_recon_only=1`, `prediction_active=0`, `whiten_active=0`, `lambda_sigreg=10`,
`lambda_var=0`, and `lambda_cov=0`. Stop on OOM or any recipe mismatch.

## 4. Launch

Create the log directory and confirm that the final checkpoint root does not already contain a run:

```bash
mkdir -p logs
test ! -e /workspace/ckpt/inv018_vjepa2_unwhitened_m512_sigreg10_only/phase1_step15000.pt
```

Launch training in the foreground inside a detached tmux session:

```bash
tmux new-session -d -s inv018_vjepa2_sigreg10 "cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder vjepa2_vitl16 --checkpoint-dir /workspace/ckpt/inv018_vjepa2_unwhitened_m512_sigreg10_only --provenance-out /workspace/ckpt/inv018_vjepa2_unwhitened_m512_sigreg10_only/run_provenance.json --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm --wandb-group inv018_vjepa2_sigreg_only --wandb-name 'Investigation 18 · Regularizer replacement · V-JEPA2 isotropy 10 only' --require-wandb 2>&1 | tee logs/inv018_vjepa2_sigreg10.log"
```

## 5. Immediate verification

```bash
tmux list-sessions
pgrep -af "python.*train.py"
nvidia-smi
tail -n 120 logs/inv018_vjepa2_sigreg10.log
tmux capture-pane -p -t inv018_vjepa2_sigreg10 -S -120
```

Verify:

- one intended `train.py` PID exists and the intended GPU is active;
- W&B uses the exact name/group above and reports a new run ID;
- the resolved encoder is V-JEPA2 at
  `b3c1679b7c34d3255ef3547f27c7b226aefab26f`;
- the resolved feature contract is `(B,1024,1024)`;
- whitening, variance, covariance, slot loss, prediction, and predicted reconstruction are off;
- SIGReg weight is 10 and its ramp is the only active geometry regularizer;
- the checkpoint and provenance paths match this guide.

Early tripwires:

- `present_recon_only=1`;
- `prediction_active=0`;
- `L_flow=0`;
- `L_recon_pred=0`;
- `whiten_active=0`;
- `grad_skipped=0`;
- finite `loss`, `L_recon`, `L_sigreg`, and `grad_norm`.

At step 0 both warmup scales are expected to start at zero. By step 2,000,
`sigreg_scale=recon_scale=1`.

## 6. Monitoring

From the local machine:

```bash
ssh hjepa-runpod 'tail -n 120 /workspace/hierarchal-jepa-flow-world-model/logs/inv018_vjepa2_sigreg10.log; nvidia-smi'
```

Do not tune or stop merely because an early scientific metric is disappointing. Stop only for a
declared tripwire, unrecoverable operational failure, or explicit human direction.
