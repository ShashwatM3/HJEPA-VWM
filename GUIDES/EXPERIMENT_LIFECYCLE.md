# Experiment lifecycle — brainstorm to KANBAN

> **Audience:** The ML engineer who owns experiments on this project.  
> **Tools:** Any AI assistant you prefer (Cursor, Codex, Bronco, etc.) for brainstorming and implementation; you drive pod deploy and launch.

---

## KANBAN folder structure (simple map)

`KANBAN/` is the living record of Phase 1 research — what we tried, measured, and learned. It is
**not** the build spec (`AGENT_FILES/AGENTS.md` and `config.py` are).

```text
KANBAN/
├── PROTOCOL.md                    # Update rules
└── PHASE_1/
    ├── README.md                  # Investigation index + all runs table
    ├── investigation_NNN/
    │   ├── DESCRIPTION.md         # Investigation question + status
    │   ├── OBSERVATIONS.md        # Cross-run synthesis
    │   ├── NEXT_STEPS.md          # What this thread spawns next
    │   ├── GUIDE.md               # Optional: how to run a sweep on a pod (investigation-level)
    │   ├── SWEEP_PLAN_*.md        # Optional: design doc (why), links to GUIDE.md (how)
    │   └── <run_slug>/
    │       ├── DESCRIPTION.md     # Hypothesis, config delta, W&B link (before launch)
    │       ├── OBSERVATIONS.md    # Triad: verdict + links to detailed readouts
    │       ├── NEXT_STEPS.md      # Next run or close
    │       ├── PLAN.md            # Implementation + execution plan (before code merge)
    │       ├── GUIDE.md           # Pod setup + exact train command (human reads this on pod)
    │       ├── METRIC_READOUT.md  # Optional: pure W&B narration (no insights)
    │       └── ANALYSIS.md        # Optional: insights derived from readout + graphs
```

**Triad rule:** every investigation and every run has `DESCRIPTION.md`, `OBSERVATIONS.md`,
`NEXT_STEPS.md`. Additional files (`GUIDE.md`, `PLAN.md`, `ANALYSIS.md`, sweep plans) are allowed
when they serve a clear role — see [`KANBAN/PROTOCOL.md`](../KANBAN/PROTOCOL.md).

**Investigation vs run folder:**

| Create a new **investigation** when… | Add a **run folder** inside an existing investigation when… |
|---|---|
| New research thread (architecture change, new regularizer family, new training mode) | Same question, different config / ablation / repeat |
| Prior investigation closed but spawned a follow-up with a new framing | Hyperparameter tick, resume, or sweep member |

When unsure, start a run inside the current investigation. Open a new investigation only when the
*question* changes materially.

---

## Lifecycle overview

```text
1. Brainstorm          →  any AI, informal
2. Plan                →  AI writes PLAN.md + run GUIDE.md (+ investigation GUIDE if sweep)
3. Implement           →  AI executes plan; engineer reviews
4. KANBAN triad        →  DESCRIPTION before launch; OBSERVATIONS/NEXT_STEPS after
5. Pod + train         →  human: NEW_POD.md → investigation GUIDE → run GUIDE
6. Analyze             →  AI: thorough W&B read (READING_EXPERIMENTS.md Q1–Q8)
7. Record              →  METRIC_READOUT.md + ANALYSIS.md → update triad OBSERVATIONS
```

---

## Stage 1 — Brainstorm

**Who:** You + your AI of choice.

**Goal:** Understand the problem, prior KANBAN results, and whether the idea was already rejected.

**Before brainstorming:**

- Skim [`KANBAN/PHASE_1/README.md`](../KANBAN/PHASE_1/README.md) for active threads.
- Read the relevant closed investigation `OBSERVATIONS.md` if reviving an old lever.
- Ask the AI to check W&B (MCP or `run_history.py`) for similar configs — **never trust memory for metric values**.

**Prompt shape:**

> We want to test whether [hypothesis]. Read investigation_XXX and check W&B for runs with
> similar config. What did we already learn? What would change in code vs CLI only?

You should leave this stage understanding the hypothesis yourself — not only the AI's summary.

---

## Stage 2 — Plan (before implementation)

**Who:** You ask the AI to produce a **thorough plan**.

**Goal:** A document detailed enough that another engineer (or future you) can implement and launch
without re-deriving decisions.

The plan must cover:

1. **Code changes** — files, functions, gradient routing, new flags, tests to add/update.
2. **Execution** — exact `train.py` command, dataset, steps, checkpoint dir, W&B group name.
3. **Success / failure criteria** — which reading cycle (A or B in [`READING_EXPERIMENTS.md`](READING_EXPERIMENTS.md)) and which gates.
4. **Risks** — what could blow up training, what prior run to compare against.

**Where the plan lives:**

| Scope | File |
|---|---|
| Single run | `KANBAN/PHASE_1/investigation_NNN/<run_slug>/PLAN.md` |
| Multi-run sweep | `KANBAN/PHASE_1/investigation_NNN/SWEEP_PLAN_<topic>.md` + per-run `PLAN.md` or a table inside the sweep plan |

**Also create before implementation:**

| File | Level | Purpose |
|---|---|---|
| `GUIDE.md` | **Run** | Copy-paste pod commands for *this* experiment (see run_041 example) |
| `GUIDE.md` | **Investigation** | Multi-GPU / multi-run launch (see investigation_007) |
| `SWEEP_PLAN_*.md` | **Investigation** | Design rationale — *why* the sweep exists (see `SWEEP_PLAN_decoder_capacity.md`) |

Examples:

- [`KANBAN/PHASE_1/investigation_011/run_041_inv011_fixed_position_present_recon/GUIDE.md`](../KANBAN/PHASE_1/investigation_011/run_041_inv011_fixed_position_present_recon/GUIDE.md)
- [`KANBAN/PHASE_1/investigation_007/GUIDE.md`](../KANBAN/PHASE_1/investigation_007/GUIDE.md)
- [`KANBAN/PHASE_1/investigation_007/SWEEP_PLAN_decoder_capacity.md`](../KANBAN/PHASE_1/investigation_007/SWEEP_PLAN_decoder_capacity.md)

**Prompt shape:**

> Create investigation_NNN/run_XXX with PLAN.md, GUIDE.md, and DESCRIPTION.md. The plan must include
> code file list, test plan, exact train command, and success criteria from READING_EXPERIMENTS cycle A.

Fill run `DESCRIPTION.md` at this stage (hypothesis, config delta vs previous run). Leave
`OBSERVATIONS.md` as a placeholder.

---

## Stage 3 — Implement

**Who:** You ask the AI to execute the plan.

**Goal:** Merged code on your branch, tests passing locally, plan and GUIDE still match the final command.

**Checklist:**

```bash
pytest -q
python -m py_compile config.py data.py models.py losses.py diagnostics.py train.py
python -c "from models import smoke_test_models; smoke_test_models()"
```

Update `PLAN.md` / `GUIDE.md` if implementation diverged from the draft plan.

**Agent coding entry:** [`AGENT_FILES/AGENTS.md`](../AGENT_FILES/AGENTS.md).

---

## Stage 4 — Deploy pod and run

**Who:** Human operator (not the agent by default).

**Read order on the pod:**

1. [`AGENT_FILES/SETUPS/NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) — system packages, pip, W&B, HF cache.
2. Investigation `GUIDE.md` — if the thread has a sweep or shared launch pattern.
3. Run `GUIDE.md` — exact smoke tests + `train.py` command for this run.

**Launch:**

```bash
tmux new -s <run_name>
cd /workspace/hierarchal-jepa-flow-world-model
git pull
# follow run GUIDE.md from here
```

Full MLOps detail: [`MLOPS.md`](MLOPS.md).

After W&B shows the run streaming, note the run ID in run `DESCRIPTION.md`.

---

## Stage 5 — Analyze (after training finishes)

**Who:** You ask the AI for an **extremely thorough** analysis.

**Requirements:**

- Use W&B MCP or `python run_history.py --run <id> --report`.
- Follow [`READING_EXPERIMENTS.md`](READING_EXPERIMENTS.md) Q1–Q8 (or present-only Q1–Q6).
- Pull real numbers at specific steps — do not invent metrics.
- Compare to the baseline named in `DESCRIPTION.md`.

**Prompt shape:**

> Analyze W&B run `<id>` using read-wandb-run skill and READING_EXPERIMENTS cycle B.
> Write METRIC_READOUT.md (facts only) and ANALYSIS.md (insights + verdict + next steps).

### Where analysis lives (two approaches)

**Approach A — single file**

| File | Contents |
|---|---|
| `ANALYSIS.md` | Full readout + interpretation + verdict in one document |

Use for smaller runs or when you want one locked-down artifact.

**Approach B — segmented (recommended for important runs)**

| File | Contents |
|---|---|
| `METRIC_READOUT.md` | **Facts only:** what each metric panel did, step by step, Q1–Q8 answers. No opinions, no "what we should try next." Lets you skip opening W&B for every panel. |
| `ANALYSIS.md` | **Insights:** what the readout implies, failure mode label, comparison to prior runs, recommended lever for the next experiment. |

Then update the **triad**:

| File | Update |
|---|---|
| `OBSERVATIONS.md` | Short synthesis + link to `ANALYSIS.md` / verdict label |
| `NEXT_STEPS.md` | Concrete next command or "close investigation" |
| Parent investigation `OBSERVATIONS.md` | Cross-run pattern if this run changes the thread |

Historical example of deep analysis:
[`KANBAN/PHASE_1/investigation_011/run_041_inv011_fixed_position_present_recon/ANALYSIS_inv011_fixed_position_present_recon.md`](../KANBAN/PHASE_1/investigation_011/run_041_inv011_fixed_position_present_recon/ANALYSIS_inv011_fixed_position_present_recon.md).

---

## Stage 6 — Close or continue

| Outcome | Action |
|---|---|
| Thread answered | Set investigation `DESCRIPTION.md` status `CLOSED`; conclusion in investigation `OBSERVATIONS.md` |
| Need another config | New run folder in same investigation; `NEXT_STEPS.md` links forward |
| New question | New `investigation_NNN/`; old investigation `NEXT_STEPS.md` points to it |

Do not delete or rewrite KANBAN history. Wrong conclusions stay with a dated correction section.

---

## AI tooling checklist

| Tool | When | Install |
|---|---|---|
| W&B MCP (`user-wandb`) | Before proposing config changes; during analysis | [W&B MCP docs](https://docs.wandb.ai/guides/hosting/mcp-server) |
| Skill `read-wandb-run` | Every run analysis | [`.claude/skills/read-wandb-run/SKILL.md`](../.claude/skills/read-wandb-run/SKILL.md) |
| Skill `wandb-primary` | Project queries, comparisons | [`.agents/skills/wandb-primary/SKILL.md`](../.agents/skills/wandb-primary/SKILL.md) |

Tell the agent explicitly:

> Use the W&B MCP server (entity `smahalanobis-uc-davis`, project `hjepa-vwm`). Never invent metric values.

---

## Quick prompt templates

**New run folder:**

> Create `KANBAN/PHASE_1/investigation_012/run_053_<slug>/` with DESCRIPTION, PLAN, GUIDE, and placeholder OBSERVATIONS/NEXT_STEPS per KANBAN/PROTOCOL.md.

**Post-run analysis:**

> Run full Reading Cycle A on W&B `<id>`. Write METRIC_READOUT.md then ANALYSIS.md in the run folder. Update OBSERVATIONS and NEXT_STEPS.

**Implement plan:**

> Execute `run_053/PLAN.md`. Match CODE_DESIGN.md and AGENTS.md invariants. Run pytest when done.
