# GUIDE — run 059 SigLIP 2 EGO4D encoder-substrate control

This reproduces the run-058 recipe on the current strict pipeline with the implemented SigLIP 2
ViT-B/16 patch encoder. It is independent of DINO. Do not copy commands from run 058's historical
guide: that guide predates the deterministic raw-data/encoder/stats/provenance contracts and its
old whitening command is no longer valid. Consequently run 058 is a historical reference, not a
causal one-delta control; causal encoder attribution needs a same-commit V-JEPA companion.

Run every section in order on the RunPod. A command returning to the prompt with no traceback is
not enough: check the stated verification. Stop at the first failed gate.

## 0. Values used by every section

Paste this block at the start of the SSH shell and paste it again after entering any new tmux
session:

```bash
cd /workspace/hierarchal-jepa-flow-world-model

export SIGLIP_REV=3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
export BATCH=64
export FRAME_MB=32
export STATS=/workspace/stats/inv016_encoder_substrate/siglip2_vitb16_ego4d_train_seed42.pt
export PREFLIGHT_DIR=/workspace/preflight/inv016_encoder_substrate
export CKPT_DIR=/workspace/ckpt/inv016_siglip2_whiten_abs_recon_cov_var_ego4d
export SMOKE_CKPT_DIR=/workspace/ckpt/inv016_siglip2_whiten_abs_recon_cov_var_ego4d_smoke
export LOG=logs/inv016_siglip2_whiten_abs_recon_cov_var_ego4d.log
export PYTHONHASHSEED=42

mkdir -p /workspace/hf_cache /workspace/stats/inv016_encoder_substrate
mkdir -p "$PREFLIGHT_DIR" "$CKPT_DIR" "$SMOKE_CKPT_DIR" logs
```

`FRAME_MB=32` is the first candidate, not permission to skip the resource test. If Section 4
settles a lower value, keep that value in the same shell and use it in every later command.

## 1. Pull and verify the experiment source

Do not discard unexplained pod changes.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short
git pull --ff-only
git rev-parse HEAD
python -m pip install -r requirements.txt
python - <<'PY'
import transformers
assert transformers.__version__ == "4.57.6", transformers.__version__
print("transformers", transformers.__version__)
PY
pytest -q
wandb login --verify
nvidia-smi
```

Verify:

- `git status --short` was empty before the pull and is empty afterward;
- the printed commit is the commit handed off with this run folder;
- all tests pass, Transformers is exactly 4.57.6, W&B verifies, and the intended GPU is idle.

## 2. Prove the real SigLIP adapter on CUDA

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

The first call downloads/caches the official checkpoint. This adapter instantiates only the
vision tower and removes the unused pooling head; there is no separate manual weight file to
find or install.

## 3. Revalidate full EGO4D completeness

Do not fit full statistics while acquisition/chunking is still changing. This command binds the
retained manifests, exact split inventories, decoded frame counts, and the Stage-4D source-UID
coverage contract. On the full corpus it can appear quiet for a while.

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
manifest, zero-frame container, or any other exception. Fixing the dataset changes its fingerprint,
so all earlier EGO4D whitening artifacts become invalid.

## 4. Settle physical batch and frame microbatch before whitening

Run the exact trainable recipe without whitening first. The full dataset identity scan happens
again; quiet startup is expected.

```bash
python train.py --resource-preflight \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$PREFLIGHT_DIR/siglip_ego4d_unwhitened_resource.json"
```

Verify the report exists and contains finite metrics, batch size 64, non-null CUDA encoder/total
peak memory, and positive examples/frames/tokens throughput.

If the encoder OOMs, retry `FRAME_MB=16`, then 8, 4, 2, and 1. Rerun Section 2 and this section
after changing it. Do not compute statistics until one value passes with headroom. If the full
B/D graph still OOMs at physical batch 64 and frame microbatch 1, stop and revise the experiment;
do not silently lower `BATCH` or add naive gradient accumulation while calling this a run-058
control.

## 5. Fit the one valid SigLIP/EGO4D whitening envelope

This is a long GPU job. Use the final `FRAME_MB` from Section 4. The 12,800 clips are selected
deterministically by clip identity; `--batch-size` controls only stats throughput.

```bash
tmux has-session -t inv016_siglip_stats 2>/dev/null \
  && tmux attach -t inv016_siglip_stats \
  || tmux new -s inv016_siglip_stats
```

Inside tmux, paste Section 0 again, restore the final `FRAME_MB` if it was changed, then run:

```bash
python whiten_stats.py \
  --data ego4d --split train \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size 8 --max-clips 12800 --seed 42 --device cuda \
  --output "$STATS"

python whiten_stats.py --inspect "$STATS"
sha256sum "$STATS"

python - "$STATS" "$SIGLIP_REV" "$FRAME_MB" <<'PY'
import json
import sys

from provenance import WHITENING_EIGENSOLVER, load_whitening_envelope

path, revision, frame_microbatch = sys.argv[1], sys.argv[2], int(sys.argv[3])
envelope = load_whitening_envelope(path)
metadata = envelope["metadata"]
spec = metadata["encoder_spec"]
summary = {
    "transform_seed": metadata["transform_seed"],
    "preprocessing_version": metadata["preprocessing_version"],
    "input_geometry": metadata["input_geometry"],
    "frame_microbatch": metadata["frame_microbatch"],
    "eigensolver": metadata["eigensolver"],
    "layout": spec["layout"],
    "feature_dim": spec["feature_dim"],
    "attention_implementation": spec["attention_implementation"],
    "cache_dir": spec["cache_dir"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
assert metadata["transform_seed"] == 42
assert metadata["input_geometry"] == [8, 256, 256]
assert metadata["frame_microbatch"] == frame_microbatch
assert metadata["eigensolver"] == WHITENING_EIGENSOLVER
assert spec["requested_revision"] == revision
assert spec["resolved_revision"] == revision
assert spec["feature_dim"] == 768
assert spec["layout"]["temporal"] == 8
assert spec["layout"]["height"] == spec["layout"]["width"] == 16
assert spec["layout"]["flatten_order"] == "time_y_x"
assert spec["layout"]["temporal_unit"] == "frame"
assert spec["frame_microbatch"] == frame_microbatch
assert spec["attention_implementation"] == "sdpa"
assert spec["cache_dir"] == "/workspace/hf_cache"
print("full whitening metadata contract OK")
PY
```

Verify:

- dataset/split are `ego4d`/`train`, transform seed is 42, clip count is exactly 12,800;
- row count is `12,800 x 2,048 = 26,214,400`;
- layout is 8x16x16 frame/time-y-x and feature dimension is 768;
- encoder family/revision/precision/frame microbatch/attention implementation/cache directory
  match this guide;
- dataset fingerprint matches Section 3 and the eigensystem is finite;
- the payload fingerprint and SHA-256 are printed.

Never reuse run 058's V-JEPA stats, an SSv2 artifact, a tiny-EGO4D artifact, or a SigLIP artifact
made with another `FRAME_MB`. Matching dimension or filename is not sufficient.

## 6. Run the whitening-active exact resource preflight

```bash
python train.py --resource-preflight \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-expected-clips 12800 --whiten-stats-path "$STATS" \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$PREFLIGHT_DIR/siglip_ego4d_whitened_resource.json"
```

Verify this passes strict stats validation and records the same feature/dataset/whitening
fingerprints. Require finite metrics, `whiten_active=1`, `present_recon_only=1`,
`prediction_active=0`, `recon_target_residual=0`, `L_flow=0`, and `L_recon_pred=0`.

## 7. Synthetic Stage 0 with the exact identity

```bash
python train.py --stage0-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-expected-clips 12800 --whiten-stats-path "$STATS" \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --checkpoint-dir "$CKPT_DIR"
```

Verify `Stage 0 sanity passed`, finite loss, frozen encoder, successful EMA transition, and the
same mode flags as Section 6. `recon_scale=0` at step 0 is expected because of warmup.

## 8. Materialize the final no-step provenance

```bash
python train.py --preflight-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-expected-clips 12800 --whiten-stats-path "$STATS" \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_ego4d_encoder_substrate \
  --wandb-name "Investigation 16 · EGO4D encoder substrate · SigLIP 2 absolute target" \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$PREFLIGHT_DIR/siglip_ego4d_exact.json"
```

Verify the JSON exists. Search it before proceeding:

```bash
grep -E 'siglip2|3f9f96cb|ego4d|12800|Investigation 16' \
  "$PREFLIGHT_DIR/siglip_ego4d_exact.json"
```

## 9. Mandatory 100-step full-EGO4D launch smoke

This uses the real dataset and full stats so strict dataset identity matches. It starts from
scratch, keeps the 2,000-step warmup, and writes to a disposable run-specific directory. It is
engineering evidence, not a scientific result.

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
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-expected-clips 12800 --whiten-stats-path "$STATS" \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_ego4d_encoder_substrate_smoke \
  --wandb-name "Investigation 16 · EGO4D encoder substrate · SigLIP 2 100-step smoke" \
  --checkpoint-dir "$SMOKE_CKPT_DIR" \
  --provenance-out "$SMOKE_CKPT_DIR/run_provenance.json" \
  --require-wandb --log-every 10 --diag-every 50 \
  2>&1 | tee logs/inv016_siglip2_ego4d_100step_smoke.log
```

Verify the process exits zero, W&B reaches step 99, metrics remain finite with no skipped/NaN
gradient, and both files exist:

```bash
test -f "$SMOKE_CKPT_DIR/phase1_step100.pt"
test -f "$SMOKE_CKPT_DIR/run_provenance.json"
sha256sum "$SMOKE_CKPT_DIR/phase1_step100.pt"
```

Do not resume the final run from this checkpoint.

## 10. Optional encoder-relative rank baseline

This does not alter training. It is useful for understanding the SigLIP substrate, but its raw
rank is not a direct score against V-JEPA.

```bash
python rank_probe.py \
  --data ego4d --split validation --probe-videos 64 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --encoder-batch 4 --cache-dtype fp16 --device cuda \
  --out-dir logs/inv016_siglip2_rank
```

Retain the JSON/plot/cache with the run evidence. Compare rank fractions and per-frame behavior,
not just unnormalized rank.

## 11. Launch the 15,000-step run

Start a persistent foreground job. Do not use `--resume`.

```bash
tmux has-session -t inv016_run059 2>/dev/null \
  && tmux attach -t inv016_run059 \
  || tmux new -s inv016_run059
```

Inside tmux, paste Section 0 again, restore the final `FRAME_MB`, confirm the final output
directory has no old checkpoints, and launch:

```bash
test -z "$(find "$CKPT_DIR" -maxdepth 1 -name 'phase1_step*.pt' -print -quit)" \
  || { echo "STOP: final checkpoint directory is not empty"; false; }

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ego4d --steps 15000 --seed 42 \
  --encoder siglip2_vitb16 --encoder-revision "$SIGLIP_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-expected-clips 12800 --whiten-stats-path "$STATS" \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv016_ego4d_encoder_substrate \
  --wandb-name "Investigation 16 · EGO4D encoder substrate · SigLIP 2 absolute target" \
  --checkpoint-dir "$CKPT_DIR" \
  --provenance-out "$CKPT_DIR/run_provenance.json" \
  --require-wandb --log-every 50 --diag-every 500 \
  2>&1 | tee "$LOG"
```

Detach with `Ctrl-b`, then `d`. The strict W&B flag makes initialization, logging, stats/provenance
artifact upload, and final checksum recording fatal instead of silently continuing offline.

## 12. Early tripwires and monitoring

After launch, reattach or open another SSH shell:

```bash
tail -n 80 "$LOG"
grep -m1 'step=0 ' "$LOG"
nvidia-smi
```

At step 0 require finite metrics plus:

```text
whiten_active = 1
present_recon_only = 1
prediction_active = 0
recon_target_residual = 0
recon_mean_norm = 0
L_flow = 0
L_recon_pred = 0
grad_skipped = 0
```

`recon_scale=0` at step 0 is expected. Stop if W&B config does not show the exact SigLIP revision,
feature fingerprint, EGO4D dataset fingerprint, whitening payload fingerprint, batch 64, and the
planned display name.

Monitor without treating init slot identity as learned geometry:

```bash
tail -f "$LOG"
watch -n 5 nvidia-smi
```

The decisive present-only panels are abstract geometry, centered slot rank, reconstruction
honesty/video gap, and stability. Raw slot rank near 32 at initialization is mechanical.

## 13. Completion handoff

After the process exits:

```bash
test -f "$CKPT_DIR/phase1_step15000.pt"
test -f "$CKPT_DIR/run_provenance.json"
sha256sum "$CKPT_DIR/phase1_step15000.pt"
tail -n 20 "$LOG"
```

Confirm W&B reached the final step and its summary contains the same final checkpoint SHA-256.
Record the W&B ID/URL in [DESCRIPTION.md](DESCRIPTION.md), then ask the coding agent to perform
Reading Cycle B with run 058 as historical context. Do not declare success from reconstruction
loss alone and do not compare the two encoders' raw reconstruction values as though they share a
target space. Before claiming an encoder-caused difference, create and run the current-commit
V-JEPA companion with its own newly validated stats/provenance.
