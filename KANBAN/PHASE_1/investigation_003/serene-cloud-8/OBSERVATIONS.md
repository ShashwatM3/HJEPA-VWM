# Observations — serene-cloud-8

## Metrics (BRIEF_V0_3)

- Slot diversity metric: looked OK
- `c_effective_rank`: **worse** than run-2
- `c_cross_video_cosine`: **~0.84** (strong video-independent collapse)
- Training: **unstable** grads

## Interpretation

**Goodhart failure.** Optimizing slot diversity does not preserve video-specific
abstract codes. Aggressive `lambda_slot` rejected.

## What worked

Nothing for production config — negative result is valuable.
