# Plan — Run 69, DINOv3 unwhitened internal-memory M=512 covariance plus variance

## Scope

Completed one-axis whitening ablation of Run 68. The DINOv3 encoder, late-projection bottleneck, decoder, present-only reconstruction, variance/covariance weights, schedule, dataset, and optimizer settings were preserved while whitening was disabled.

## Configuration delta from Run 68

| Field | Run 68 | Run 69 |
|---|---|---|
| `whiten_features` | `true` | `false` |
| `whiten_stats_path` | DINOv3/EGO4D artifact | empty |
| whitening payload fingerprint | required | `null` |
| run name/group/paths | Run 68 identity | Run 69 identity |

All architecture and loss settings remained matched, including projection order `in_proj -> latent_blocks -> abstract_proj`, `lambda_var=0.5`, and `lambda_cov=0.01`.

## Acceptance evidence

Stage 0 passed with `whiten_active=0.0`, `present_recon_only=1.0`, `prediction_active=0.0`, `grad_skipped=0.0`, and `instability_warn=0.0`. The full run then completed successfully as W&B `fiactcw6`, and the final step-15000 checkpoint checksum was verified. Upload verification recorded `7` checkpoint objects (`2.9 GiB`) under `s3://hjepa-volume/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/` and `1` preflight object under `s3://hjepa-volume/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var/`.

## Outcome

The final reconstruction snapshot (`L_recon=0.11770`, `L_recon_present=0.13343`) is consistent with whitening contributing to the degradation qualitatively reported for Run 68. This is not a precise causal estimate because Run 68's exact metrics remain unavailable in the repository.
