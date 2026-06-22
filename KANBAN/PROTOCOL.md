# KANBAN — Agent protocol

This folder records **what happened** in Phase 1 research runs. It is not a
build spec. Authoritative phase definitions live in `AGENT_FILES/PHASES/`.

---

## The triad rule

Every **investigation** and every **run** has exactly three markdown files:

| File | Role |
|---|---|
| `DESCRIPTION.md` | Question, status, context |
| `OBSERVATIONS.md` | Evidence, belief evolution, conclusions |
| `NEXT_STEPS.md` | What to do next, or what this spawned |

Do **not** create additional files inside investigation or run folders (no
`HYPOTHESES.md`, `RUNS.md`, postmortems, charts, etc.). Hypothesis history
belongs in `OBSERVATIONS.md` as dated sections.

---

## When to update which file

### Starting a new investigation

1. Create `PHASE_1/investigation_NNN/` with the three files.
2. Fill `DESCRIPTION.md` first (question, status `OPEN`, parent links).
3. Leave `OBSERVATIONS.md` minimal until a run produces data.
4. Fill `NEXT_STEPS.md` with the first concrete experiment.

### Launching a run

1. Create `investigation_NNN/<wandb-run-name>/` (or a short slug if no W&B name).
2. Fill run `DESCRIPTION.md` **before** launch (hypothesis, command, config delta).
3. Leave run `OBSERVATIONS.md` empty or placeholder until results arrive.

### After a run completes

1. Update the run's `OBSERVATIONS.md` with metrics and interpretation.
2. Update the run's `NEXT_STEPS.md` (next run, close, or spawn).
3. Update the parent investigation's `OBSERVATIONS.md` with cross-run synthesis.
4. Update the parent investigation's `NEXT_STEPS.md`.

### Closing an investigation

1. Set status to `CLOSED` in investigation `DESCRIPTION.md`.
2. Write a **Conclusion** section in investigation `OBSERVATIONS.md`.
3. Point investigation `NEXT_STEPS.md` at follow-up investigations (by folder name).

---

## Context-loading discipline

When picking up work:

1. Read `KANBAN/PHASE_1/README.md` for current status.
2. Read **only** the active investigation's three files.
3. Read the specific run folder being worked on (if any).
4. Do **not** load other investigations unless cross-linking is required.

For architecture and acceptance gates, read `AGENT_FILES/PHASES/PHASE_1.md` —
point to it; do not duplicate it here.

---

## Writing discipline

- **When in doubt, write less.** New investigations are cheap; bloated files are expensive.
- If a finding does not fit the triad, open a new investigation instead of a new file.
- **Do not delete or rewrite history.** Wrong conclusions stay in `OBSERVATIONS.md`
  with a later correction section explaining what changed.
- Investigations and runs may **overlap, branch, or inform each other.** Capture
  relationships with markdown links, not directory nesting.
- Run folder names use the **W&B display name** when known (e.g. `cerulean-snow-13`).
  Use a short slug only when no W&B name exists in the record.

---

## Phase docs vs KANBAN

| Source | Purpose |
|---|---|
| `AGENT_FILES/PHASES/PHASE_{1,2,3,4}.md` | What each project phase **is** and what to build |
| `KANBAN/PHASE_1/` | What we **tried**, **measured**, and **learned** in Phase 1 so far |

Phase 2 adds fine flow; Phase 3 adds pixels; Phase 4 adds multi-horizon coarse
prediction. KANBAN only tracks Phase 1 research until Phase 2 opens its own folder.
