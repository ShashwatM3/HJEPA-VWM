# Investigation 001 — Can Phase 1 train without numerical blow-up?

**Status:** CLOSED  
**Opened:** 2026-06-09 (first full Phase 1 launch on RunPod)  
**Closed:** 2026-06-10 (postmortem + hyperparameter retune landed in code)

## Question

Can the Phase 1 training loop (frozen encoder, AdamW, bf16, gradient clipping)
run for the planned step budget without gradient explosion, NaN weights, or
process crash?

## Why it matters

Every later experiment (collapse fixes, horizon sweeps, acceptance gates) is
wasted if the base recipe is unstable. This investigation gates whether Phase 1
is runnable at all.

## Parent context

- Spec: [`AGENT_FILES/PHASES/PHASE_1.md`](../../AGENT_FILES/PHASES/PHASE_1.md)
- Branched from: initial Phase 1 implementation (commit `5cb8330` / `e1e1956`)
- Spawned: [investigation_002](investigation_002/) (throughput), [investigation_003](investigation_003/) (collapse visible in Run 1 metrics before crash)

## Runs in this investigation

| Run | Role |
|---|---|
| [`peachy-terrain-5`](peachy-terrain-5/) | First full launch — failed at step ~10750 |
