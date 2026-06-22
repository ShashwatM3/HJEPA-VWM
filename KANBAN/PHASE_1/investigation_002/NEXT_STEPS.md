# Next steps — Investigation 002

Investigation **closed** after [`charmed-haze-4`](charmed-haze-4/OBSERVATIONS.md) validated selective decode.

## Why no further work here

Throughput was a dataloader issue, not a model issue. Full SSv2 runs in
[investigation_003](investigation_003/) and [investigation_005](investigation_005/)
proceeded on this fix.

## Spawned

**Next expensive training:** [investigation_001](investigation_001/) → `peachy-terrain-5` (already run at closure time).

If s/step regresses on new hardware → contingency
[`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/) (not opened).
