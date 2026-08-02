# Next Steps - investigation_001

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

The project moved from one-off long-run optimism into deliberate smoke, diagnostic, and collapse-control runs.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: The long baseline was not a success signal. It exposed late instability and weak prediction, which made optimizer hardening, restart discipline, and clearer acceptance metrics necessary before interpreting longer runs.
- Runs covered: 005.

## Follow-Up Chain

This investigation feeds into `investigation_002`: RunPod, dataloader, W&B, and tiny-data smoke validation. The reason is: Before treating losses or diagnostics as research evidence, the project needed to prove that the pod, W&B sync, SSv2 tiny dataset, and basic training loop worked end to end.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — Investigation 001

Investigation **closed** after [`peachy-terrain-5`](run_005_peachy-terrain-5/).

## Why follow-ups exist

Run 1 proved the loop can train and learn (`L_flow` fell) but **not** stably to
completion (gradient explosion) and **not** with healthy `c_t` (rank ~5). Stability
fixes went into code; collapse became the next question.

## Spawned

1. **[investigation_002](../investigation_002/)** — throughput had to be fixed before
   burning GPU on full SSv2 (closed; selective decode).
2. **[investigation_003](../investigation_003/)** — why `c_effective_rank` stays ~5;
   entry run [`exalted-lion-6`](../investigation_003/run_006_exalted-lion-6/).

**Guardrail carried forward:** sustained `grad_skipped` → stop the run (codified in
[investigation_005](../investigation_005/) after `elated-snowflake-15`).
