# KANBAN — Execution plan for getting Phase 1 to a real result

> **Audience:** any coding agent picking up the operations work on this project.
> **Companion audience:** the human operator (who has the only SSH access to
> the RunPod pod, the only RunPod console login, and the eyes on W&B).
>
> **Scope:** the operations work that happens *between* Phase 1 implementation
> (done) and Phase 2 (not started). This folder is **not** the project's
> Phase 1/2/3/4 — those live in [`../PHASES/`](../PHASES/). These are
> **execution phases of the plan** to actually run the project's Phase 1 to
> completion.

---

## Why this folder exists

Phase 1 code is implemented and smoke-tested (commit `4c1abb3` on
`phase1-v0.2-frozen-encoder`). What remains is operational:

1. The dataloader is currently CPU-bound — current throughput is ~1.66 s/step,
   which extrapolates to ~14 hours of A100 time for the canonical 30k-step
   Phase 1 run.
2. We need to actually launch the 30k run and measure the **acceptance gate**
   (`coarse_vs_copy_ratio ≤ 0.70` after step 10k, per
   [`../PHASES/PHASE_1.md`](../PHASES/PHASE_1.md) §12).
3. We have a contingency for further CPU→GPU offload, only if needed.

---

## Folder layout

```
KANBAN/
├── README.md                              ← this file
├── 01-OPTIMIZE-DATALOADER/                ← do first; cheap, big win
│   ├── DETAILED_UNDERSTAND.md
│   ├── TASKS.md                           ← coding agent's checklist
│   └── HUMAN_TASKS.md                     ← human operator's checklist
├── 02-LAUNCH-FULL-PHASE-1-RUN/            ← the real Phase 1 training run
│   ├── DETAILED_UNDERSTAND.md
│   ├── TASKS.md
│   └── HUMAN_TASKS.md
└── 03-CPU-TO-GPU-OFFLOAD/                 ← contingency; only if 01 isn't enough
    ├── DETAILED_UNDERSTAND.md
    ├── TASKS.md
    └── HUMAN_TASKS.md
```

## Execution order

| Folder | When to execute |
|---|---|
| `01-OPTIMIZE-DATALOADER/` | Now — before any long training run. |
| `02-LAUNCH-FULL-PHASE-1-RUN/` | After 01 is verified (s/step measurably lower). |
| `03-CPU-TO-GPU-OFFLOAD/` | **Only if** 01 leaves s/step > ~0.7. Otherwise skip. |

## Files in each folder

### `DETAILED_UNDERSTAND.md`
Context, motivation, success goals, risks. Read this first when entering a
folder; do not jump straight to `TASKS.md`.

### `TASKS.md` — for the coding agent
Pure agent-executable tasks: code edits, file creation, local shell commands
(git, syntax checks), commits, pushes. Each task has:

- **Status:** `[NOT DONE]` or `[DONE]`
- **Description:** one-paragraph summary
- **Instructions:** the exact commands or code edits to perform
- **How to verify:** the concrete success check before marking `[DONE]`

### `HUMAN_TASKS.md` — for the human operator
Pure human-executable tasks: SSH into the pod, run commands on the pod,
check W&B in the browser, redeploy a pod tier on RunPod, etc. Same
status / description / instructions / verify format as `TASKS.md`.

## Agent ↔ human handoff convention

The agent and the human cannot work in parallel on the same task because most
project actions live on the RunPod pod (SSH-only, human-controlled). The
workflow is strictly **handoff** based:

1. **Agent works through `TASKS.md` in order.** Some `TASKS.md` tasks end
   with an explicit blocker marker:

   ```
   > **⏸ PAUSE — BLOCKED ON HUMAN.**
   > Wait for the human to complete HUMAN_TASKS.md task H<N> and report
   > back <specific piece of information>. Do not proceed to the next task.
   ```

   When the agent hits such a marker, it stops and waits.

2. **Human works through `HUMAN_TASKS.md` in order.** Each `HUMAN_TASKS.md`
   task ends with a **`Report back to the agent:`** section listing the
   exact piece(s) of information the agent needs to unblock.

3. **Human resumes the agent** by sending those reported values back in chat.
   The agent verifies, then continues from where it paused.

4. **Both files share task numbering**: `TASKS.md` uses `A1.1`, `A2.1`, etc.
   (the "A" stands for "Agent"); `HUMAN_TASKS.md` uses `H1`, `H2`, etc. Each
   file cross-references the matching items in the other file so the handoff
   points are obvious.

This means **neither file is complete on its own** — they are read together.
A task marked `[DONE]` in `TASKS.md` is only really done once the
corresponding `HUMAN_TASKS.md` items have also been completed and reported.
