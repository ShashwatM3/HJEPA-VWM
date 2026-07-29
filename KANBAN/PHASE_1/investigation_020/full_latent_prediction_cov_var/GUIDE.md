# Guide — full-latent prediction arm

Use the environment variables and all gates from [`../GUIDE.md`](../GUIDE.md).

The exact scientific mode is:

```text
--temporal-target full_latent
```

Launch command:

```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data ego4d \
  --encoder "$INV020_ENCODER" \
  --n-c "$INV020_N_C" \
  --d-c "$INV020_D_C" \
  --bottleneck-mixer-dim "$INV020_M" \
  --warm-start-from "$INV020_SOURCE_CHECKPOINT" \
  --temporal-target full_latent \
  --checkpoint-dir /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/full_latent \
  --provenance-out /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/full_latent/run_provenance.json \
  --wandb-entity smahalanobis-uc-davis \
  --wandb-project hjepa-vwm \
  --wandb-group inv020_pretrained_bottleneck_temporal_target \
  --wandb-name "Investigation 20 · Temporal target · Full-latent prediction, DINOv3 $INV020_N_C slots by $INV020_D_C" \
  --require-wandb
```

Run inside tmux with `set -o pipefail` and tee to `logs/inv020_full_latent.log`, as shown in the
shared guide.

Immediate arm-specific assertions:

- `predict_residual=false`;
- flow target is detached future `c_plus`;
- flow noise is unit Gaussian;
- predicted endpoint is a full future latent, with no `c_t` add-back;
- `lambda_recon_pred=0`, so future reconstruction remains diagnostic-only;
- copy means use the present latent as the future estimate.
