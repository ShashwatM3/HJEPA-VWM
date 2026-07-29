# Guide — residual prediction arm

Use the environment variables and all gates from [`../GUIDE.md`](../GUIDE.md).

The exact scientific mode is:

```text
--temporal-target residual
```

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data ego4d \
  --encoder "$INV020_ENCODER" \
  --n-c "$INV020_N_C" \
  --d-c "$INV020_D_C" \
  --bottleneck-mixer-dim "$INV020_M" \
  --warm-start-from "$INV020_SOURCE_CHECKPOINT" \
  --temporal-target residual \
  --checkpoint-dir /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/residual \
  --provenance-out /workspace/ckpt/inv020_pretrained_bottleneck_temporal_target/residual/run_provenance.json \
  --wandb-entity smahalanobis-uc-davis \
  --wandb-project hjepa-vwm \
  --wandb-group inv020_pretrained_bottleneck_temporal_target \
  --wandb-name "Investigation 20 · Temporal target · Residual prediction, DINOv3 $INV020_N_C slots by $INV020_D_C" \
  --require-wandb
```

Run inside tmux with `set -o pipefail` and tee to `logs/inv020_residual.log`, as shown in the shared
guide.

Immediate arm-specific assertions:

- `predict_residual=true`;
- target is the detached `B_EMA` future-minus-present residual;
- residual noise scale is finite and positive;
- prediction-side diagnostic endpoint adds the predicted residual to online `c_t`;
- `lambda_recon_pred=0`, so that endpoint is diagnostic-only for reconstruction;
- copy means predict zero residual.
