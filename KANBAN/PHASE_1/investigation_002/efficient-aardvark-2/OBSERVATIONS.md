# Observations — efficient-aardvark-2

## Outcome

**Baseline captured.** Confirmed CPU-bound decode bottleneck and metric parity with
earlier local runs.

## Evidence

- **~1.66 s/step** wall-clock on A100 pod
- CPU saturated (~7.8× core-time vs real time); GPU waiting on batches
- Step-0 `L_flow` / `L_var` bit-identical to pre-pod smoke within batch noise
- Root cause identified: `_read_video_decord` decoded **all** frames then sliced to 16

## Interpretation

Throughput problem isolated to dataloader, not model. Fix target: decode only needed
frame indices.
