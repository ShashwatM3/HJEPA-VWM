# PROTOCOL.md — Agent Operating Procedure

> **Audience:** AI coding agents working on HJEPA-VWM Phase 1.
> **Read [`AGENT_FILES/AGENTS.md`](../AGENTS.md) on first clone, then this file at the start of every session before touching code.**
>
> **Living docs:** Which files you must keep current, when, and how → **`AGENTS.md` §2.2**.
> Which files are human-owned → **`AGENTS.md` §2.1**.

---

## 0. Repository layout (agent docs)

Entry point for new agents: **[`AGENT_FILES/AGENTS.md`](../AGENTS.md)** (implementation-grounded architecture, shapes, training step, invariants, doc precedence).

```
AGENT_FILES/
├── AGENTS.md                         ← start here (living code reference)
├── AGENT-BEHAVIOUR/
│   ├── PROTOCOL.md                   ← this file
│   ├── CODE_DESIGN.md                ← naming, docstrings, detach rules
│   └── WORKFLOW.md                   ← redirect → GUIDES/EXPERIMENT_LIFECYCLE.md
├── KNOWLEDGE/                        ← redirect stubs only (see README there)
└── SETUPS/
    ├── SETUP.md
    ├── NEW_POD.md
    └── VOLUME_LAYOUT.md

GUIDES/                               ← human-facing playbooks + project knowledge
├── README.md                         ← index
├── latest_brief.md                   ← architecture narrative (NOT ground truth)
├── original_brief.pdf
├── CODEBASE_STRUCTURE.md
├── PROBLEMS_METRICS_AND_EXPERIMENTS.md
├── EXPERIMENT_LIFECYCLE.md
├── MLOPS.md
└── READING_EXPERIMENTS.md

KANBAN/PHASE_1/                       ← experiment history (what we tried)
```

When any doc names another file, use the **full path from repo root** (e.g. `GUIDES/latest_brief.md`).

---

## 1. Document map — what each file is for

| Path | Role | When to read |
|---|---|---|
| `AGENT_FILES/AGENTS.md` | **Living code reference** — shapes, modules, training step, defaults, invariants, **briefs are not ground truth**. | First clone; before any code change. |
| `GUIDES/CODEBASE_STRUCTURE.md` | **File map** — which root file to open for a given task. | When orienting to the repo layout. |
| `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md` | **Metric glossary + experiment problem history** — grounded in `diagnostics.py` / `train.py`. | Before interpreting runs or adding diagnostics. |
| `GUIDES/latest_brief.md` | **Architecture narrative** (v0.3) — past intent and empirical notes. **Not an override for code.** | Optional context before architecture changes. |
| `GUIDES/original_brief.pdf` | **Historical baseline** — initial design. | When tracing how intent evolved. |
| `GUIDES/READING_EXPERIMENTS.md` | W&B reading cycles Q1–Q8. | When analyzing or writing run verdicts. |
| `GUIDES/EXPERIMENT_LIFECYCLE.md` | Brainstorm → KANBAN workflow. | When scoping runs or analysis artifacts. |
| `GUIDES/MLOPS.md` | RunPod, W&B, volumes. | When producing pod commands. |
| `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` | Code style — flat files, naming, docstrings. | Before creating/renaming files. |
| `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md` | RunPod volume paths. | Before paths, data, or subsets. |
| `KANBAN/PHASE_1/README.md` | Run index — what was already tried. | Before proposing configs. |
| `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` (this file) | How you operate. | Every session, first. |

**Precedence when documents conflict:** see `AGENT_FILES/AGENTS.md` §2. **Code and tests beat briefs.**

---

## 2. Operating mode

1. **Phase 1 code exists.** Your job is usually a **targeted change** or **experiment support**, not greenfield scaffolding.
2. **Trace code before docs** when behavior is ambiguous.
3. **One scope per session.** Do not quietly implement later phases unless the human requests it.
4. **Runnable deliverables.** Keep `pytest -q` and smoke tests passing.
5. **Research is allowed.** Propose new mechanisms when evidence supports them; briefs do not veto sound ideas.

---

## 3. Deployment target — RunPod

**Canonical volume reference:** [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../SETUPS/VOLUME_LAYOUT.md).

Repo at `/workspace/hierarchal-jepa-flow-world-model/`; `data/`, `checkpoints/`, `ssv2_raw/`, and `hf_cache/` as siblings under `/workspace/`. Override data root locally with `JEPA_DATA_ROOT` only.

**Workflow:** local dev → git push → SSH ([`SETUP.md`](../SETUPS/SETUP.md) Path B) → `git pull`. Fresh pod: [`NEW_POD.md`](../SETUPS/NEW_POD.md). Detail: [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md).

---

## 4. Research permission

You **may** search papers and official docs when APIs or formulas are ambiguous.

- If research contradicts **code** or **locked invariants** (`AGENTS.md` §15), stop and ask the human.
- Prefer: code → `AGENTS.md` → `GUIDES/PROBLEMS_METRICS…` → KANBAN → briefs (context only) → papers.

You **must not** substitute a different architecture because a blog post suggested it.

---

## 5. Ask-the-human protocol

**Must ask:** changes to §15 invariants, locked dimensions, gradient routing, ambiguous gradient paths, missing volume data.

**May proceed with documented assumption:** `num_workers`, W&B naming, minor logging.

---

## 6. Shape-contract discipline

Every public tensor function: docstring per `CODE_DESIGN.md` §4. Cross-check `AGENTS.md` §4 and `config.py`.

---

## 7. Stop-gradient hygiene

Centralize detaches in `as_target()`. See `AGENTS.md` §15 and `CODE_DESIGN.md` §6.

---

## 8. Bypass-test mindset

Phase 1 diagnostics must exist for components you touch. Thresholds: `GUIDES/READING_EXPERIMENTS.md` and `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`.

---

## 9. Session checklist

```
[ ] 1. Read AGENTS.md (if first session).
[ ] 2. Read this file (PROTOCOL.md).
[ ] 3. Read CODE_DESIGN.md before editing code.
[ ] 4. Read VOLUME_LAYOUT.md if touching paths or data.
[ ] 5. Skim KANBAN + GUIDES/PROBLEMS_METRICS if changing architecture/losses/metrics.
[ ] 6. Trace the hot path in code; run targeted tests when done.
[ ] 7. Update agent-maintained living docs (AGENTS.md §2.2) if the change requires it.
```

---

## 10. Tooling and quality bar

`black` (line 100), `ruff`, type hints, W&B logging per `config.py` cadence.

---

## 11. What "done" means for a code change

1. Contract preserved or updated with tests.
2. `pytest -q` and smoke tests pass.
3. **Agent-maintained living docs updated** if your change touched what they describe —
   see [`AGENTS.md`](../AGENTS.md) §2.2 (implementation, ops, KANBAN). Do not edit
   human-owned docs (§2.1) unless asked.
4. No unexplained `.detach()`.

Report: what changed, gradient paths affected, which living docs you updated, verify commands.
