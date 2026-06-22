# Next steps — Investigation 001

Investigation **closed** after [`peachy-terrain-5`](peachy-terrain-5/).

## Why follow-ups exist

Run 1 proved the loop can train and learn (`L_flow` fell) but **not** stably to
completion (gradient explosion) and **not** with healthy `c_t` (rank ~5). Stability
fixes went into code; collapse became the next question.

## Spawned

1. **[investigation_002](investigation_002/)** — throughput had to be fixed before
   burning GPU on full SSv2 (closed; selective decode).
2. **[investigation_003](investigation_003/)** — why `c_effective_rank` stays ~5;
   entry run [`exalted-lion-6`](investigation_003/exalted-lion-6/).

**Guardrail carried forward:** sustained `grad_skipped` → stop the run (codified in
[investigation_005](investigation_005/) after `elated-snowflake-15`).
