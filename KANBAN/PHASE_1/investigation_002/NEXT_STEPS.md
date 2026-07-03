# Next Steps - investigation_002

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

With infrastructure usable, the research shifted to full-data Phase 1 collapse and baseline diagnostics.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success.
- Runs covered: 001, 002, 003, 004.

## Follow-Up Chain

This investigation feeds into `investigation_003`: collapse diagnostics, variance floor, horizon, and slot-loss variants. The reason is: The early runs made it clear that lowering L_flow was not enough. This investigation tested whether variance, horizon, and slot/covariance choices could produce a video-specific abstract latent and a useful predictor.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — Investigation 002

Investigation **closed** after [`charmed-haze-4`](run_004_charmed-haze-4/OBSERVATIONS.md) validated selective decode.

## Why no further work here

Throughput was a dataloader issue, not a model issue. Full SSv2 runs in
[investigation_003](../investigation_003/) and [investigation_005](../investigation_005/)
proceeded on this fix.

## Spawned

**Next expensive training:** [investigation_001](../investigation_001/) → `peachy-terrain-5` (already run at closure time).

If s/step regresses on new hardware → contingency
[`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD) (not opened).
