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
- Related to [investigation_001](investigation_001/) (same era) but independent question
- Contingency if insufficient: [`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/) — **not needed**

## Runs in this investigation

No W&B runs recorded. Validation was a timed 200-step smoke (`commit 4c1abb3` baseline vs `5e78caa` after fix) documented in KANBAN 01.
