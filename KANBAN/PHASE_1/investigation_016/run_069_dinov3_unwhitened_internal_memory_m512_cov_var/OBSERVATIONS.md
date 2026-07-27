# Observations — Run 69, DINOv3 unwhitened internal-memory M=512 covariance plus variance

## Status

COMPLETED — full 15,000-step run finished successfully.

## Completion identity

- W&B: [`fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6)
- Final checkpoint: `/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/phase1_step15000.pt`
- Final checkpoint SHA-256: `fbd4bda4c75780c06b35c837a1291218c4525c1549baf978ef3561eaf81e7cb8`
- Uploaded successfully:
  - `s3://hjepa-volume/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/`
    (`7` objects, `2.9 GiB`)
  - `s3://hjepa-volume/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var/`
    (`1` object)

## Verified final metrics

| Metric | Final value |
|---|---:|
| `L_recon` | 0.11770 |
| `L_recon_present` | 0.13343 |
| `L_recon_shuffled_c` | 0.17960 |
| `L_recon_video_gap` | 0.04617 |
| `L_var` | 0.00111 |
| `L_cov` | 0.30737 |
| `L_slot` | 0.00639 |
| `L_sigreg` | 0.00734 |
| `L_flow` | 0 |
| `L_recon_pred` | 0 |

`L_flow=0` and `L_recon_pred=0` are consistent with the documented present-only configuration; they do not constitute forecasting evidence.

## Qualified whitening comparison

Run 69's final reconstruction (`L_recon=0.11770`, `L_recon_present=0.13343`) is consistent with whitening contributing to the reconstruction degradation qualitatively reported for Run 68. This conclusion is deliberately limited: Run 68's exact metrics are not verified in the repository, so the available evidence does not support a precise effect size or a stronger causal claim.

The final correct-vs-shuffled reconstruction gap is `0.04617`. No encoder-level cross-video cosine value was supplied in the completion evidence, and the current training diagnostics do not log one; see the repository audit in the task handoff rather than adding an unsupported value here.
