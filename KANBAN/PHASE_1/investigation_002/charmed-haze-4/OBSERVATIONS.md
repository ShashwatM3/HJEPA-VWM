# Observations — charmed-haze-4

## Outcome

**Fix validated.** Throughput improved; metrics unchanged within batch noise.

## Evidence

- Lower s/step vs `efficient-aardvark-2` (improvement ~15%, not the hoped ~3×)
- Step-0 / step-50 `L_flow`, `L_var`, `grad_has_nan` match pre-fix baseline
- Remaining cost: decord seeking through VP9 keyframes, not array allocation

## Interpretation

Selective decode was sufficient to proceed with full SSv2 runs. CPU→GPU offload
([`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/))
was **not** opened.
