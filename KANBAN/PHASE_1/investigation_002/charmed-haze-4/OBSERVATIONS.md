# Observations — charmed-haze-4

## Outcome

**Fix validated.** Throughput improved; metrics unchanged within batch noise.

## Evidence

- **`real 4m42.195s` for 200 steps = 1.41 s/step**, vs `efficient-aardvark-2`'s
  `5m31.226s = 1.66 s/step` pre-fix — a ~15% improvement, well short of the hoped ~3×
  that decoding 16/≈60 frames might have suggested.
- Step-0 `L_flow=2.8726348876953125`, `L_var=0.42669573426246643`, `grad_has_nan=0` —
  **bit-identical** to the pre-fix `efficient-aardvark-2` step-0 line (and step-50/100/150
  match within batch-sampling noise). This is the proof the refactor was
  behavior-preserving: decoding only needed frame indices did not change what the model sees.
- Remaining cost is decord **seeking through VP9 keyframes** (`.webm` at `num_threads=1`),
  not numpy array allocation — so the cheap win was already captured and the rest is
  Plan Phase 03 (CPU→GPU decode) territory.

## Interpretation

Selective decode was sufficient to proceed with full SSv2 runs. CPU→GPU offload
([`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/))
was **not** opened.
