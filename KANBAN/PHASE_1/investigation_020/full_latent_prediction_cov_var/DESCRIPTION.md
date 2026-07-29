# Full-latent prediction from a reconstruction-pretrained bottleneck

## Status

READY — DINOv3 source selected; warm-start and temporal-target implementation complete.

W&B ID: pending.

## Hypothesis

With a useful present bottleneck already learned, the current full-latent rectified-flow target may
benefit from predicting the whole future code because static and changing components remain in one
target distribution. The pretrained condition may remove some of the representation-learning
burden that harmed historical full-latent runs.

The falsifier is explicit: if copy loss falls while the copy ratio stays near or above one, the
pretrained bottleneck is becoming temporally static and `F_c` is not forecasting.

## Source identity

Locked to the selected Investigation-019 DINOv3 arm:

| Field | Value |
|---|---|
| selected shape | `N_c=64`, `D_c=512`, `M=512` |
| source W&B ID | `60yaqw6d` |
| source checkpoint | `/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt` |
| checkpoint SHA-256 | `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1` |
| encoder fingerprint | `963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415` |
| dataset fingerprint | `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c` |

## Config delta

Relative to the locked Investigation-020 common recipe:

```text
predict_residual = false
```

All other settings and initialization must match the residual arm.

## Intended W&B identity

```text
group: inv020_pretrained_bottleneck_temporal_target
name: Investigation 20 · Temporal target · Full-latent prediction, DINOv3 64 slots by 512
```

## Comparator

[`../residual_prediction_cov_var/`](../residual_prediction_cov_var/) from the same git SHA and
source checkpoint.
