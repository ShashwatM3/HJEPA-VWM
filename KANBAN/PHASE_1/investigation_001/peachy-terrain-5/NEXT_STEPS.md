# Next steps — peachy-terrain-5

## Why this run stops here

Gradient explosion at ~step 10750 (peak LR too aggressive on tiny data). Separately,
`c_effective_rank` stuck at ~5 — collapse visible before the crash. Post-explosion
checkpoints are unusable.

## What changed in code (not a new W&B run)

Hyperparameter retune landed in repo: halved peak LRs, `grad_clip=0.5`, grad-skip
guard at norm > 50, budget 30k → 15k for tiny. See
`AGENT_FILES/KANBAN/02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md`.

## Spawned

**Next investigation — collapse:** [investigation_003](../investigation_003/) — rank ~5
and cross-video cosine must be understood before trusting any long run.

**First run in that arc:** [`exalted-lion-6`](../investigation_003/exalted-lion-6/) —
cheap 500-step diagnostic on `ssv2_tiny` to confirm collapse instrumentation before
full-SSv2 spend.

**Parallel prerequisite (already closed):** [investigation_002](../investigation_002/) —
dataloader throughput; had to be sufficient before full SSv2.

This run **closed investigation 001**. No resume from this run's checkpoints.
