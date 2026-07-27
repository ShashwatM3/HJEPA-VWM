# Observations — Run 68, DINOv3 whitened internal-memory M=512 covariance plus variance

## Status

LAUNCHED — qualitative observations are available, but repository-visible evidence does not establish completion, an exact W&B ID, checkpoint identity, or final metric values.

## Preliminary qualitative report

- Whitening was confirmed to remain enabled.
- Reconstruction loss was reported to worsen to approximately `0.3`.
- Effective rank of `C` improved.
- Cross-video cosine decreased.
- The next controlled test is unwhitened Run 69.

Treat these as preliminary qualitative observations. Do not promote them to exact final values or a completed verdict without W&B/checkpoint evidence. Attribution across the broader experiment sequence is confounded because whitening and the late `M -> D_c` projection were both present. Run 69 isolates whitening by preserving the projection order and the `lambda_var=0.5`, `lambda_cov=0.01` settings while disabling whitening.

## Pending evidence

- W&B ID/URL and terminal state.
- DINOv3 whitening-envelope path, payload fingerprint, and file SHA-256.
- Final checkpoint path and SHA-256; committed provenance artifact.
- Exact final and steps 12,000–14,500 late-window metrics.
- Present-only Reading Cycle B verdict.

Do not invent missing values.