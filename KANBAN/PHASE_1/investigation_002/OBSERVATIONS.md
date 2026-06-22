# Observations — Investigation 002 (dataloader throughput)

## Baseline (pre-fix)

200-step smoke on `ssv2_tiny` (commit `4c1abb3`):

- **~1.66 s/step** wall-clock
- CPU saturated (~7.8× core-time vs real time); GPU waiting on batches
- Root cause: `_read_video_decord` decoded **all** frames then sliced to 16 needed

## Fix

Decode only the 16 frame indices required per clip (`commit 5e78caa`). Bit-identical
frames to the old path. Kept `num_threads=1` for VP9 `.webm` reliability.

## Belief update

Throughput was a **data-loader issue**, not a model issue. Fix was sufficient to
proceed with full SSv2 runs; investigation 03 CPU→GPU offload was **not** opened.

## Conclusion

**CLOSED — solved by selective decode.** No run-level W&B artifact; pod-side
timing confirmed improvement (target ≤0.8 s/step per KANBAN 01 goals).

## Cross-investigation

Enabled the expensive runs in [investigation_003](investigation_003/) and [investigation_005](investigation_005/) on full SSv2.
