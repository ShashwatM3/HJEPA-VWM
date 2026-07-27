# GUIDE — Run 66, unwhitened EGO4D M=512 with covariance plus variance

This guide launches one fresh 15,000-step present-only run. It is Run 64 with exactly the settled
geometry bundle activated: `lambda_var=0.5`, `lambda_cov=0.01`. Do not resume Run 64.

## 1. Required identity

```text
data = ego4d
seed = 42
present_recon_only = true
prediction_active = 0
whiten_features = false
bottleneck_mixer_dim = 512
n_c = 32
d_c = 256
lambda_recon = 1.0
lambda_var = 0.5
lambda_cov = 0.01
lambda_sigreg = 0
lambda_slot = 0
```

W&B:

```text
name  = Investigation 16 · Raw-feature geometry · M=512 covariance plus variance
group = inv016_unwhitened_m512_geometry
```

Durable outputs:

```text
tmux      = inv016_m512_cov_var
log       = logs/inv016_unwhitened_m512_cov_var.log
checkpoint= /workspace/ckpt/inv016_unwhitened_m512_cov_var
provenance= /workspace/ckpt/inv016_unwhitened_m512_cov_var/run_provenance.json
preflight = /workspace/preflight/inv016_unwhitened_m512_cov_var/ego4d_m512_cov_var_resource.json
```

## 2. Pull and verify the published implementation

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short --branch
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull --ff-only origin phase1-v0.2-frozen-encoder
git status --short --branch
git rev-parse HEAD
python -m pip install -r requirements.txt
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
wandb login --verify
```

Require a clean tracked worktree, exact local/remote SHA parity, passing tests/smokes, and working
W&B authentication. No whitening-stat prerequisite exists; do not pass any `--whiten-*` flag.

## 3. Shared exact arguments

Every gate and the paid run use this scientific recipe:

```bash
--data ego4d
--steps 15000
--seed 42
--encoder vjepa2_vitl16
--encoder-revision b3c1679b7c34d3255ef3547f27c7b226aefab26f
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
--lambda-var 0.5
--lambda-cov 0.01
--lambda-sigreg 0
--lambda-slot 0
--bottleneck-latent-blocks 3
--bottleneck-mixer-dim 512
--decoder-dim 512
--decoder-blocks 4
--n-c 32
--lr-bottleneck 1e-4
--lr-coarse-flow 1e-4
--lr-decoder 1e-4
--log-every 50
--diag-every 500
```

## 4. Stage 0 and resource preflight

Run Stage 0 with the shared arguments and `--stage0-only`. Require finite metrics,
`present_recon_only=1`, `prediction_active=0`, `whiten_active=0`, `L_flow=0`,
`L_recon_pred=0`, `grad_skipped=0`, and finite positive `L_var`/`L_cov`.

Then run `--resource-preflight` with the same shared arguments plus:

```bash
--checkpoint-dir /workspace/ckpt/inv016_unwhitened_m512_cov_var_preflight
--provenance-out /workspace/preflight/inv016_unwhitened_m512_cov_var/ego4d_m512_cov_var_resource.json
```

The preflight must exit zero and report finite loss/gradients, non-null CUDA peak memory, and
positive throughput at batch 64. Full EGO4D provenance can be quiet and CPU-bound while scanning
video headers; inspect process I/O before treating it as stalled.

## 5. Exact paid launch

Launch in foreground inside detached tmux, with unbuffered output and `set -o pipefail`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
  --data ego4d --steps 15000 --seed 42 \
  --encoder vjepa2_vitl16 \
  --encoder-revision b3c1679b7c34d3255ef3547f27c7b226aefab26f \
  --encoder-precision bf16 --encoder-frame-microbatch 8 \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size 64 --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-latent-blocks 3 --bottleneck-mixer-dim 512 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --log-every 50 --diag-every 500 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_unwhitened_m512_geometry \
  --wandb-name "Investigation 16 · Raw-feature geometry · M=512 covariance plus variance" \
  --checkpoint-dir /workspace/ckpt/inv016_unwhitened_m512_cov_var \
  --provenance-out /workspace/ckpt/inv016_unwhitened_m512_cov_var/run_provenance.json \
  --require-wandb \
  2>&1 | tee logs/inv016_unwhitened_m512_cov_var.log
```

Do not launch if the tmux name, paid checkpoint directory, paid log, or a matching training
process already exists without first inspecting it. Do not overwrite or implicitly resume.

## 6. Early tripwires

Verify the session and exact training PID, GPU activity, W&B ID/name/group, checkpoint destination,
and the first diagnostic row. Stop only this run if any of these scientific tripwires fails:

- wrong dataset, width, code size, loss mode, or geometry weights;
- whitening or prediction becomes active;
- `L_flow` or `L_recon_pred` becomes nonzero;
- any NaN/nonfinite loss, skipped update spiral, or persistent instability warning;
- W&B initialization fails under `--require-wandb`;
- OOM or checkpoint/provenance path collision.

## 7. Completion proof

Completion requires all of:

```text
process exits 0
W&B state = finished
final expected training/diagnostic history exists
/workspace/ckpt/inv016_unwhitened_m512_cov_var/phase1_step15000.pt exists
/workspace/ckpt/inv016_unwhitened_m512_cov_var/run_provenance.json exists
checkpoint SHA-256 recorded locally and in W&B summary
resolved config/provenance matches Section 1
```

After completion, pull the full W&B history, apply Reading Cycle B, and update this run folder plus
the parent investigation and Phase 1 index. Preserve the single-source diagnostic limitation.
