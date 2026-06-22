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

## W&B smoke runs

| Run | Steps | Finding |
|---|---|---|
| `youthful-pond-1` | 100 | Pod + W&B + training loop OK |
| `efficient-aardvark-2` | 200 | ~1.66 s/step pre-fix; metrics bit-identical |
| `charmed-haze-4` | 200 | Post-fix faster; metrics unchanged |

Source: [`AGENT_FILES/COMPLETE_FULL_CHAT`](../../AGENT_FILES/COMPLETE_FULL_CHAT) (lines ~482–2810).

## Conclusion

**CLOSED — solved by selective decode.** Pod-side timing confirmed improvement
(target ≤0.8 s/step per KANBAN 01 goals not fully hit, but sufficient to proceed).

## Cross-investigation

Enabled the expensive runs in [investigation_003](investigation_003/) and [investigation_005](investigation_005/) on full SSv2.
