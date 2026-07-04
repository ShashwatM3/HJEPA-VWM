# KANBAN — Agent protocol

This folder records **what happened** in Phase 1 research runs. It is not a
build spec. Authoritative implementation reference: `AGENT_FILES/AGENTS.md` and `config.py`.

Operator playbooks (lifecycle, MLOps, W&B reading cycles): [`GUIDES/`](../GUIDES/README.md).

---

## The triad rule

Every **investigation** and every **run** has exactly three markdown files:

| File | Role |
|---|---|
| `DESCRIPTION.md` | Question, status, context |
| `OBSERVATIONS.md` | Evidence, belief evolution, conclusions |
| `NEXT_STEPS.md` | What to do next, or what this spawned |

**Additional allowed files** (see [`GUIDES/EXPERIMENT_LIFECYCLE.md`](../GUIDES/EXPERIMENT_LIFECYCLE.md)):

| File | Level | Role |
|---|---|---|
| `PLAN.md` | Run | Implementation + execution plan (before merge) |
| `GUIDE.md` | Run or investigation | Pod setup + exact launch commands (human reads on pod) |
| `SWEEP_PLAN_*.md` | Investigation | Design doc — *why* a sweep exists |
| `METRIC_READOUT.md` | Run | Pure W&B metric narration (no insights) |
| `ANALYSIS.md` | Run | Insights derived from readout + graphs |

Do **not** create other ad-hoc markdown files (no `HYPOTHESES.md`, `RUNS.md`, postmortems,
etc.). Hypothesis history belongs in `OBSERVATIONS.md` or `ANALYSIS.md` as dated sections.

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
3. Fill run `PLAN.md` and `GUIDE.md` before implementation / launch (see `GUIDES/EXPERIMENT_LIFECYCLE.md`).
4. Leave run `OBSERVATIONS.md` empty or placeholder until results arrive.

### After a run completes

1. Write `METRIC_READOUT.md` and/or `ANALYSIS.md` in the run folder (see `GUIDES/EXPERIMENT_LIFECYCLE.md`).
2. Update the run's `OBSERVATIONS.md` with verdict and links to analysis.
3. Update the run's `NEXT_STEPS.md` (next run, close, or spawn).
4. Update the parent investigation's `OBSERVATIONS.md` with cross-run synthesis.
5. Update the parent investigation's `NEXT_STEPS.md`.

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

For architecture and acceptance gates, read `AGENT_FILES/AGENTS.md` and
`GUIDES/READING_EXPERIMENTS.md` — point to them; do not duplicate them here.

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
| `AGENT_FILES/AGENTS.md` + `config.py` | What Phase 1 **is in code** — modules, defaults, invariants |
| `GUIDES/latest_brief.md` | Architecture **narrative** and empirical notes (not ground truth) |
| `KANBAN/PHASE_1/` | What we **tried**, **measured**, and **learned** in Phase 1 so far |

Phase 2 adds fine flow; Phase 3 adds pixels; Phase 4 adds multi-horizon coarse
prediction. KANBAN only tracks Phase 1 research until Phase 2 opens its own folder.

---

## Code hygiene

Keep the production path clean. When writing or modifying training code, regularizers,
or diagnostics:

- Prefer **flags over hard-coded constants** for any empirical knob.
- Default flags to **current behavior**, so the baseline is one switch away and
  reproducible (`--lambda-var`, `--lr-coarse-flow`, AGC all follow this).
- **Bake fixes in, gate experiments.** Settled fixes become defaults with the flag
  removed (e.g. init `62b94dd`); only live experimental knobs stay as switches.
- **No dead code.** When an investigation rejects a mechanism, remove the abandoned
  regularizer and unused branches (slot loss was rejected — do not leave it wired into
  the loss by default).
- When in doubt, write less code.

---

## Internal cross-reference

Before introducing a new mechanism, read the relevant investigation(s) and the code to
learn why the current approach was chosen. The KANBAN exists to stop re-litigating
settled questions. If a hypothesis was rejected and recorded in an
`OBSERVATIONS.md` (e.g. slot-diversity loss as a training objective, H3 in
investigation_003), **do not silently resurrect it without new evidence** — cite the
prior result and state what is different now.

---

## External research

When a problem is not addressed by the existing investigations or codebase, research
external techniques (papers, established architectures, OSS). Propose a technique only
if either: (a) it is demonstrated effective on the **specific** problem we face, or
(b) it enhances the architecture while addressing the open issue. Report the **source,
the evidence, and the risk** before proposing implementation (as was done for the
slot-diversity penalty: Perceiver / Slot-Attention collapse, "Trap of Mediocrity").

---

## Empirical grounding via W&B

Use the W&B MCP server (project `hjepa-vwm`, entity `smahalanobis-uc-davis`) whenever a
hypothesis or fact-check benefits from real run data:

- Before proposing a config change, **check whether a similar config was already tried**
  and what happened (e.g. λ_cov=0.0027 ran four times as a slot adjunct — never isolated).
- When brainstorming next steps, **pull recent run trajectories** rather than trusting
  what the chat or files claim.
- When debugging an unexpected metric, **verify the trajectory in W&B** before drawing
  conclusions (post-AGC grad norm hid the royal-cherry cliff; the cliff was at 8600,
  not 8500).
- **Never invent metric values from memory — pull them.**
