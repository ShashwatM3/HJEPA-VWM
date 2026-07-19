# GUIDE — Run 66, SigLIP 2 unwhitened internal-memory M=512 (EGO4D)

This is the exact launch guide for the single 15,000-step SigLIP 2 encoder-substrate run. It is the
V-JEPA M=512 arm ([`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/),
W&B [`4biwq87o`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o)) with **only the
frozen encoder swapped** from `vjepa2_vitl16` to `siglip2_vitb16`. There is **no whitening** and
**no variance/covariance/SIGReg/slot regularizer** — identical to the internal-memory sweep recipe.

Run every section in order on the RunPod. A command returning to the prompt with no traceback is not
enough: check the stated verification. Stop at the first failed gate. Do not use `--resume`.

## 0. Fresh-pod bootstrap first

If this pod is fresh, complete only Sections 1–5 of
[`AGENT_FILES/SETUPS/NEW_POD.md`](../../../AGENT_FILES/SETUPS/NEW_POD.md): system packages, cache +
checkpoint paths, clone + `git checkout phase1-v0.2-frozen-encoder`, install `requirements.txt`, and
`wandb login`. Do not run the generic SSv2 examples after Section 5. Then continue here.

## 1. Values used by every section

Paste this block at the start of the SSH shell, and again after entering any new tmux session:

```bash
cd /workspace/hierarchal-jepa-flow-world-model

export SIGLIP_REV=3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
export BATCH=64
export FRAME_MB=32
export PREFLIGHT_DIR=/workspace/preflight/inv016_encoder_substrate_m512
export CKPT_DIR=/workspace/ckpt/inv016_siglip2_unwhitened_memory_m512
export SMOKE_CKPT_DIR=/workspace/ckpt/inv016_siglip2_unwhitened_memory_m512_smoke
export LOG=logs/inv016_siglip2_unwhitened_memory_m512.log
export PYTHONHASHSEED=42

mkdir -p /workspace/hf_cache /workspace/ckpt "$PREFLIGHT_DIR" "$CKPT_DIR" "$SMOKE_CKPT_DIR" logs
export HF_HOME=/workspace/hf_cache
```

`FRAME_MB=32` is only a starting candidate. It is an encoder throughput knob and does **not** change
the scientific recipe or the encoder output. Section 4 settles the value that fits memory; keep that
same value in the shell for every later command.

## 2. Pull and verify the experiment source

Do not discard unexplained pod changes.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short --branch
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull --ff-only origin phase1-v0.2-frozen-encoder
git rev-parse HEAD
python -m pip install -r requirements.txt
python - <<'PY'
import transformers
assert transformers.__version__ == "4.57.6", transformers.__version__
print("transformers", transformers.__version__)
PY
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
wandb login --verify
nvidia-smi
```

Verify: `git status --short` is empty before and after the pull; the printed commit is the commit
handed off with this run folder; all tests pass; Transformers is exactly `4.57.6`; W&B verifies; and
the intended GPU is idle.

## 3. Prove the real SigLIP 2 adapter on CUDA

```bash
python encoders.py --smoke \
  --encoder siglip2_vitb16 \
  --revision "$SIGLIP_REV" \
  --precision bf16 \
  --frame-microbatch "$FRAME_MB" \
  --attention-implementation sdpa \
  --hf-cache-dir /workspace/hf_cache \
  --batch-size 1 \
  --device cuda \
  | tee "$PREFLIGHT_DIR/siglip_adapter_smoke.json"
```

Verify the JSON says:

```text
requested_revision = resolved_revision = 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
transformers_version = 4.57.6
parameter_count = 85843200
trainable_parameter_count = 0
output_shape = [1, 2048, 768]
output_finite = true
device = cuda
```

The first call downloads/caches the official checkpoint. This adapter instantiates only the vision
tower and removes the unused pooling head; there is no separate manual weight file.

## 4. Revalidate full EGO4D completeness

Binds the retained manifests, split inventories, decoded frame counts, and the Stage-4D source-UID
coverage contract. On the full corpus it can appear quiet for a while (CPU-bound header scan).

```bash
python - <<'PY'
from config import Config
from provenance import build_dataset_identity

cfg = Config()
cfg.data.dataset = "ego4d"
identity = build_dataset_identity(cfg, require_complete=True)
assert identity.get("completeness") == "stage-4d-verified", identity.get("completeness")
print("dataset:", identity["dataset"])
print("train clips:", identity["splits"]["train"]["count"])
print("validation clips:", identity["splits"]["validation"]["count"])
print("fingerprint:", identity["fingerprint"])
print("completeness:", identity["completeness"])
PY
```

Stop on a missing/extra source UID, split leakage, chunk-manifest mismatch, missing retained
manifest, zero-frame container, or any other exception.

## 5. Settle physical batch and frame microbatch (resource preflight)

Run the exact trainable recipe. The full dataset-identity scan happens again; quiet startup is
expected. This is the **whole recipe** — no whitening flags, all regularizer weights zero.

```bash
python train.py --resource-preflight \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --lambda-var 0 --lambda-cov 0 --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$PREFLIGHT_DIR/ckpt" \
  --provenance-out "$PREFLIGHT_DIR/siglip_ego4d_m512_resource.json"
```

Verify the report exists and contains finite metrics, batch size 64, non-null CUDA encoder/total
peak memory, and positive examples/frames/tokens throughput.

If the encoder OOMs, retry with `FRAME_MB=16`, then `8`, `4`, `2`, `1` (rerun Section 3 and this
section after changing it). Keep the first value that passes with headroom for every later command.
Do **not** lower `BATCH` or add gradient accumulation — that would break the run-064 control.

## 6. Synthetic Stage 0 with the exact identity

```bash
python train.py --stage0-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --lambda-var 0 --lambda-cov 0 --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$CKPT_DIR"
```

Verify `Stage 0 sanity passed`, finite loss, frozen encoder, successful EMA transition,
`present_recon_only=1`, `whiten_active=0`, and `prediction_active=0`. `recon_scale=0` at step 0 is
expected because of warmup.

## 7. Materialize the final no-step provenance

```bash
python train.py --preflight-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --lambda-var 0 --lambda-cov 0 --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_unwhitened_memory \
  --wandb-name "Investigation 16 · Encoder substrate · SigLIP 2 unwhitened memory M=512" \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$PREFLIGHT_DIR/siglip_ego4d_m512_exact.json"
```

Search it before proceeding:

```bash
grep -E 'siglip2|3f9f96cb|ego4d|Investigation 16' "$PREFLIGHT_DIR/siglip_ego4d_m512_exact.json"
```

## 8. (Optional) 100-step full-EGO4D launch smoke

Engineering evidence, not a scientific result. Starts from scratch, keeps the 2,000-step warmup,
writes to a disposable directory. Do not resume the final run from this checkpoint.

```bash
test -z "$(find "$SMOKE_CKPT_DIR" -maxdepth 1 \
  \( -name 'phase1_step*.pt' -o -name 'run_provenance.json' \) -print -quit)" \
  || { echo "STOP: smoke checkpoint directory is not empty; use a fresh path"; false; }

set -o pipefail
python train.py \
  --data ego4d --steps 100 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --lambda-var 0 --lambda-cov 0 --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_unwhitened_memory_smoke \
  --wandb-name "Investigation 16 · Encoder substrate · SigLIP 2 unwhitened memory M=512 100-step smoke" \
  --checkpoint-dir "$SMOKE_CKPT_DIR" \
  --provenance-out "$SMOKE_CKPT_DIR/run_provenance.json" \
  --require-wandb --log-every 10 --diag-every 50 \
  2>&1 | tee logs/inv016_siglip2_m512_100step_smoke.log

test -f "$SMOKE_CKPT_DIR/phase1_step100.pt"
sha256sum "$SMOKE_CKPT_DIR/phase1_step100.pt"
```

Verify the process exits zero, W&B reaches step 99, metrics remain finite with no skipped/NaN
gradient.

## 9. Launch the 15,000-step run

Start a persistent foreground job inside tmux. Do not use `--resume`.

```bash
tmux has-session -t inv016_run066 2>/dev/null && echo "STOP: session exists; inspect it" \
  || tmux new-session -d -s inv016_run066 "bash -lc '
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
export PYTHONHASHSEED=42
test -z \"\$(find /workspace/ckpt/inv016_siglip2_unwhitened_memory_m512 -maxdepth 1 -name phase1_step*.pt -print -quit)\" || { echo STOP: final checkpoint dir not empty; exit 9; }
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab \
  --encoder-precision bf16 --encoder-frame-microbatch '\"\$FRAME_MB\"' \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size 64 --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --lambda-var 0 --lambda-cov 0 --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-mixer-dim 512 --bottleneck-latent-blocks 3 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_encoder_substrate_unwhitened_memory \
  --wandb-name \"Investigation 16 · Encoder substrate · SigLIP 2 unwhitened memory M=512\" \
  --checkpoint-dir /workspace/ckpt/inv016_siglip2_unwhitened_memory_m512 \
  --provenance-out /workspace/ckpt/inv016_siglip2_unwhitened_memory_m512/run_provenance.json \
  --require-wandb --log-every 50 --diag-every 500 \
  2>&1 | tee logs/inv016_siglip2_unwhitened_memory_m512.log
'"
```

> Note: substitute the settled `FRAME_MB` from Section 5 if you did not export it in the launching
> shell. `--require-wandb` makes init/logging/provenance/final-checksum failures fatal instead of
> silently continuing offline.

## 10. Early tripwires and monitoring

```bash
tmux list-sessions
pgrep -af '[p]ython.*train.py'
nvidia-smi
tail -n 100 "$LOG"
grep -m1 'step=0 ' "$LOG"
```

At the first logged step require finite metrics plus:

```text
data                = ego4d
present_recon_only  = 1
prediction_active   = 0
whiten_active       = 0
recon_target_residual = 0
n_c                 = 32
bottleneck_mixer_dim = 512
lambda_recon        = 1.0
lambda_var = lambda_cov = lambda_sigreg = lambda_slot = 0
L_flow              = 0
L_recon_pred        = 0
grad_skipped        = 0
```

`recon_scale=0` at step 0 is expected (warmup). Stop if the W&B config does not show the exact
SigLIP revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`, the SigLIP feature fingerprint
(`D_e=768`, `N_e=2048`, layout `8x16x16`), the EGO4D dataset fingerprint, batch 64, and the planned
display name.

Ongoing:

```bash
tail -f "$LOG"
watch -n 5 nvidia-smi
tmux capture-pane -p -t inv016_run066 -S -160
```

## 11. Completion proof

After the process exits:

```bash
test -f "$CKPT_DIR/phase1_step15000.pt"
test -f "$CKPT_DIR/run_provenance.json"
sha256sum "$CKPT_DIR/phase1_step15000.pt"
tail -n 20 "$LOG"
```

Confirm W&B reached step 15000, its state is `finished`, and its summary contains the same final
checkpoint SHA-256. Then record the W&B ID/URL and SHA-256 in [`DESCRIPTION.md`](DESCRIPTION.md) and
[`OBSERVATIONS.md`](OBSERVATIONS.md), run present-only Reading Cycle B, and update the parent
investigation records.

## 12. Interpretation guardrail

Do **not** declare an "encoder winner" from raw reconstruction loss: V-JEPA and SigLIP have
different `D_e`/`N_e`/layout, so their cosine targets differ. Compare within-run diagnostics
(correct-vs-shuffled gap, exact-chunk conditioned share, geometry trajectory, stability) against the
V-JEPA M=512 control ([`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/),
W&B `4biwq87o`). The fixed EGO4D validation batch is within-source/exact-chunk, so all
"cross-video"/shuffled numbers are exact-chunk claims, not global cross-source claims.
