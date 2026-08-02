# Mission

## Outcome

Be able to reconstruct and explain HJEPA-VWM from first principles, including:

- every implemented module, its exact inputs, outputs, dimensions, initialization, parameters, and
  role;
- the disk-to-loss data path and every stop-gradient boundary;
- all training modes and how each changes compute and gradient flow;
- optimizer groups, clipping, LR/EMA/loss schedules, logging cadence, and checkpoint cadence;
- dataset creation, deterministic sampling, augmentation, feature whitening, diagnostics, probes,
  W&B, remote RunPod operation, and recovery boundaries;
- the empirical history that explains why the architecture looks the way it does;
- the planned coarse-to-fine and pixel stages, clearly separated from implemented Phase 1.

The practical bar is: if a tech lead chooses an arbitrary tensor, parameter, flag, metric, artifact,
or remote-run transition, the learner can state where it came from, its shape or value, where it
goes, what can update it, and what failure would reveal a broken contract.

## Starting point

The learner knows Python and machine-learning basics but should not need prior familiarity with this
repository. The course grounds notation before using it and builds from one clip to the complete
experiment lifecycle.

## Evidence of mastery

Mastery is demonstrated when the learner can, without opening the code:

1. draw the implemented Phase 1 graph and the planned full graph;
2. trace all tensors and gradients for five modes: default prediction, residual prediction,
   present-only reconstruction, residual reconstruction, and whitening;
3. derive the default parameter counts and compression ratios;
4. calculate temporal overlap for an arbitrary horizon `k`;
5. compute LR, EMA, ramp, log, diagnostic, and checkpoint behavior at a named step;
6. explain every Q1–Q8 diagnostic verdict and the limitations of the fixed EGO4D validation batch;
7. describe strict resume validation and why a checkpoint is not a complete experiment backup;
8. launch, monitor, and close a run using the documented remote protocol without conflating
   bootstrap authorization with paid-run authorization.

## Course sequence

1. Truth labels, notation, and system-level map.
2. Data, encoders, bottleneck, target branch, flow, and decoder.
3. Objectives, gradients, optimization, and schedules.
4. Diagnostics, provenance, datasets, probes, and remote operations.
5. Research history and future architecture.
6. Retrieval drills, whiteboard reconstruction, and spaced review.

## Guardrails

- Current code beats generic ML intuition.
- Tests define executable invariants.
- KANBAN records experiments, not build truth.
- Old architecture briefs describe intent and history, not current execution.
- Raw reconstruction losses are not comparable across encoder feature spaces.
- Present-only success is not future-prediction success.
- A run is not complete merely because a process exists or W&B has a page.
