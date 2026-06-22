# Observations — init-fixes-full-ssv2-run-2

## Metrics (from BRIEF_V0_3 / chat record)

- `c_effective_rank`: rose to **~9** then **stalled**
- `c_cross_video_cosine`: **~0.72** (video-agnostic drift)
- Gradients: stable (skip guard not firing)

## Interpretation

Init + full data **helped but did not solve** dimensional collapse. Confirmed
collapse is primarily a **training-objective** issue (variance floor too weak),
not init alone.

## Surprises

Rank improved vs peachy (~5) without changing `lambda_var` — init matters as
hygiene but plateaus below acceptance targets.
