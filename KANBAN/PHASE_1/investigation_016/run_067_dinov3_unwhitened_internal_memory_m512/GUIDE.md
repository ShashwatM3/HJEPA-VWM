# GUIDE — Run 67, DINOv3 unwhitened internal-memory M=512 (EGO4D)

> **Local-label correction (2026-07-27):** W&B `it7sq8nz` is canonical local
> [Run 069](../run_069_unwhitened_dinov3_m512_no_geometry_regularizers/). This retained guide
> records the source-branch launch and must not be treated as a separate Run 067.

This records the exact successful 15,000-step DINOv3 encoder-substrate launch. It follows the
run-066 SigLIP 2 guide with the encoder identity and collision-avoiding operational paths changed.
There is no whitening and no variance/covariance/SIGReg/slot regularizer. Do not use `--resume`.

## 1. Verified values

```bash
cd /workspace/hierarchal-jepa-flow-world-model

export DINO_REV=5931719e67bbdb9737e363e781fb0c67687896bc
export BATCH=64
export FRAME_MB=32
export CKPT_DIR=/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512
export LOG=logs/inv016_dinov3_unwhitened_memory_m512.log
export PYTHONHASHSEED=42
export HF_HOME=/workspace/hf_cache
```

Source identity:

```text
branch = codex/task3-dino-run066
commit = 083cf8a6e87168702efe46ac6bfe485756dcb439
```

## 2. Exact 15,000-step training command

The successful run used no resume checkpoint.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
  --data ego4d --steps 15000 --seed 42 \
  --encoder dinov3_vitb16 --encoder-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --encoder-precision bf16 --encoder-frame-microbatch 32 \
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
  --wandb-name "Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512" \
  --checkpoint-dir /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512 \
  --provenance-out /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/run_provenance.json \
  --require-wandb --log-every 50 --diag-every 500 \
  2>&1 | tee logs/inv016_dinov3_unwhitened_memory_m512.log
```

## 3. Completion proof

```bash
test -f "$CKPT_DIR/phase1_step15000.pt"
test -f "$CKPT_DIR/run_provenance.json"
sha256sum "$CKPT_DIR/phase1_step15000.pt"
tail -n 20 "$LOG"
```

Verified result:

```text
W&B run ID       = it7sq8nz
W&B state        = finished
training steps   = 15000
checkpoint SHA   = f4513c9ce01e18f4db6cc6c3434c5a5f53749019f36992ed332fc14e67dd33db
provenance       = /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/run_provenance.json
W&B artifact     = it7sq8nz-provenance:v0
artifact type    = run-provenance
artifact state   = COMMITTED
```

Checkpoints were written at steps `2500, 5000, 7500, 10000, 12500, 15000`. The model checkpoint
is retained on the RunPod persistent volume and was not uploaded to W&B.

## 4. Interpretation guardrail

Do not declare an encoder winner from raw reconstruction loss. DINOv3 and V-JEPA have different
feature spaces. Compare correct-vs-shuffled gap, exact-chunk conditioned share, geometry trajectory,
and stability against the V-JEPA M=512 control
([`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/),
W&B `4biwq87o`). The fixed EGO4D validation batch is within-source/exact-chunk, so shuffled metrics
do not establish global cross-source conditioning.
