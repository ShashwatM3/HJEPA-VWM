# Investigation 002 — Is dataloader throughput sufficient for full SSv2?

**Status:** CLOSED  
**Opened:** 2026-06 (before / parallel to first long runs)  
**Closed:** 2026-06 (decode-only-needed-frames landed, commit `5e78caa`)

## Question

Is CPU-side video decode the bottleneck for Phase 1 training, and can we fix it
without changing what the model sees?

## Why it matters

At ~1.66 s/step, a 15k–30k run is prohibitively expensive and the A100 sits idle.
Full SSv2 experiments are impractical until throughput improves.

## Parent context

- Operational plan: [`AGENT_FILES/KANBAN/01-OPTIMIZE-DATALOADER/`](../../AGENT_FILES/KANBAN/01-OPTIMIZE-DATALOADER/)
- Related to [investigation_001](../investigation_001/) (same era) but independent question
- Contingency if insufficient: [`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/) — **deferred, not needed yet**

## Runs in this investigation

Smoke runs only — no dedicated training run. Four W&B smokes (all `ssv2_tiny`):
`youthful-pond-1` (connectivity), `efficient-aardvark-2` (1.66 s/step pre-fix baseline),
`comfy-glade-3` (`--log-every 1` dense smoke, pre-fix), `charmed-haze-4` (1.41 s/step
post-fix). Timed throughput checks, not training experiments. See OBSERVATIONS.md.
