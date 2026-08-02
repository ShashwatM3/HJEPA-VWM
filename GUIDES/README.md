# Guides

Human-facing playbooks and project knowledge for HJEPA-VWM research. **Start here** if you are
new to the project or running experiments.

---

## Operator playbooks

| Guide | Who | What |
|---|---|---|
| [**EXPERIMENT_LIFECYCLE.md**](EXPERIMENT_LIFECYCLE.md) | ML engineer | End-to-end loop: brainstorm → plan → implement → pod → train → analyze → KANBAN |
| [**MLOPS.md**](MLOPS.md) | ML engineer | RunPod, W&B, volumes, logs, `run_history.py` |
| [**READING_EXPERIMENTS.md**](READING_EXPERIMENTS.md) | Engineer + agent | Q1–Q8 W&B reading cycles for full-prediction and present-only runs |

---

## Project knowledge (architecture, code map, metrics)

These docs are for **humans and agents** who need context beyond the training loop. They are **not**
immutable specs — see [`AGENT_FILES/AGENTS.md`](../AGENT_FILES/AGENTS.md) §2 for how briefs relate
to code.

| Document | What it covers |
|---|---|
| [**latest_brief.md**](latest_brief.md) | Current architecture narrative (v0.3): hierarchy intent, empirical Phase 1 notes, common CLI overrides. **Historical thinking, not ground truth.** |
| [**original_brief.pdf**](original_brief.pdf) | Initial hierarchical JEPA-Flow design (v0.1 baseline). |
| [**CODEBASE_STRUCTURE.md**](CODEBASE_STRUCTURE.md) | File map: training code, tests, MLOps scripts, doc layout. **Living** — update when repo structure changes. |
| [**PROBLEMS_METRICS_AND_EXPERIMENTS.md**](PROBLEMS_METRICS_AND_EXPERIMENTS.md) | Metric glossary (grounded in code) + experiment problem history. **Living** — update when metrics or major failure modes change. |

---

## Related docs outside this folder

| Document | Role |
|---|---|
| [`AGENT_FILES/AGENTS.md`](../AGENT_FILES/AGENTS.md) | Agent entry: implementation-grounded architecture, shapes, invariants |
| [`KANBAN/PHASE_1/README.md`](../KANBAN/PHASE_1/README.md) | What we tried (Phase 1 run index) |
| [`KANBAN/PROTOCOL.md`](../KANBAN/PROTOCOL.md) | KANBAN update rules |
| [`AGENT_FILES/SETUPS/NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) | Fresh pod bootstrap |
| [`AGENT_FILES/SETUPS/SETUP.md`](../AGENT_FILES/SETUPS/SETUP.md) | First RunPod deploy / repeat deploy |
| [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../AGENT_FILES/SETUPS/VOLUME_LAYOUT.md) | `/workspace` path contract |
