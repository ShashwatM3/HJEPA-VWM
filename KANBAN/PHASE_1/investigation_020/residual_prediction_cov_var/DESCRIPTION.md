# Residual prediction from a reconstruction-pretrained bottleneck

## Status

RUNNING — launched from clean commit `7649f8efde1b104dd81cfbd110af18d499f67304`.

W&B: [`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t).

## Hypothesis

With a useful present bottleneck already learned, predicting the detached EMA temporal residual
will focus `F_c` on change rather than spending capacity transporting the static component of the
future latent. The run should therefore improve the late copy ratio relative to the matched
full-latent arm.

The falsifier is explicit: if `coarse_copy_loss` rises but `coarse_vs_copy_ratio` stays near one,
the bottleneck is dynamic but `F_c` is again predicting approximately zero residual.

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

Relative to the locked Investigation-020 common recipe, the only difference is:

```text
predict_residual = true
```

All regularizers, reconstruction settings, data, shape, initialization, optimizer schedule, and
diagnostics must match the full-latent arm.

## Intended W&B identity

```text
group: inv020_pretrained_bottleneck_temporal_target
name: Investigation 20 · Temporal target · Residual prediction, DINOv3 64 slots by 512
```

## Comparator

[`../full_latent_prediction_cov_var/`](../full_latent_prediction_cov_var/) from the same git SHA and
source checkpoint.
