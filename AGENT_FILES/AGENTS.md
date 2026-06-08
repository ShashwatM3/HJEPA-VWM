# AGENTS.md — Coding agent entry point

> **Audience:** Any AI coding agent that clones or opens this repository.
> **Purpose:** Replace generic repo guessing with a focused map of the project, the filesystem, and the documents you must read before writing code.

---

## 1. What this repository is

**HJEPA-VWM** (Hierarchical JEPA-Flow Video World Model, v0) is a **video world model** that learns a **two-level latent hierarchy**:

- **Abstract latent `c_t`** — future-relevant structure (32 tokens × 256 dims).
- **Detailed latent `e_t`** — texture and local visual detail (256 context tokens × 384 dims).

The model predicts **future latents**, not pixels directly. Two **flow-matching** networks (`F_c`, `F_e`) learn velocity fields from noise to future targets. Targets come from an **EMA target branch** (always stop-gradient). A **frame generator `D`** (Stage 4 only, after latent hierarchy is verified) renders pixels in frozen VAE latent space.

This is **not** a video diffusion model. The compressed predictive state is the point; bypass tests prove the hierarchy is real.

**Current repo state (planning phase):** There is **no Python implementation yet**. The repository contains agent instructions, architecture comprehension, phased build specs, and human RunPod setup guides. Implementation is produced by an agent executing `PHASES/PHASE_1.md` → `PHASE_2.md` → `PHASE_3.md` after a human completes `SETUPS/SETUP.md`.

**Training target:** RunPod GPU pod with a network volume. Dataset: Something-Something V2 (SSv2). Code is edited locally → git push → SSH → `git pull` → train on pod.

---

## 2. Mandatory read order (do this before any code)

| Order | File | Why |
|---|---|---|
| **1** | [`AGENT-BEHAVIOUR/PROTOCOL.md`](AGENT-BEHAVIOUR/PROTOCOL.md) | **How you must behave** — phase execution, escalation, stop-gradient rules, acceptance gates, RunPod layout. Read at the **start of every session**. |
| **2** | [`AGENT-BEHAVIOUR/CODE_DESIGN.md`](AGENT-BEHAVIOUR/CODE_DESIGN.md) | **How you must write code** — flat 5–6 file layout, naming map, docstrings, `as_target()` pattern, black/ruff. Read before creating or renaming any file. |
| **3** | [`KNOWLEDGE/UNDERSTANDING.md`](KNOWLEDGE/UNDERSTANDING.md) §0–§3, §2.6, §6 | **What you are building** — shapes, locked constants, modules, stop-gradient table. Re-read §2, §2.6, §6 whenever touching latents or losses. |
| **4** | [`KNOWLEDGE/hierarchical_jepa_flow_architecture_brief.pdf`](KNOWLEDGE/hierarchical_jepa_flow_architecture_brief.pdf) | **Authoritative spec** from the tech lead. Consult sections cited by the active phase doc. |
| **5** | [`PHASES/PHASE_<N>.md`](PHASES/PHASE_1.md) | **What to build this session** — only after the human says "execute Phase N." Follow the workflow table top-to-bottom. |

**Do not read** generic tutorials or substitute a different architecture. **Do not start coding** until steps 1–2 are done and step 5 is opened.

Human operators (not agents) use [`SETUPS/SETUP.md`](SETUPS/SETUP.md) and [`SETUPS/SETUP_POD.md`](SETUPS/SETUP_POD.md) for first-time RunPod setup and troubleshooting.

---

## 3. Document precedence (when sources conflict)

1. `KNOWLEDGE/hierarchical_jepa_flow_architecture_brief.pdf` — architecture intent.
2. `KNOWLEDGE/UNDERSTANDING.md` §2.6 — numerical constants.
3. Active `PHASES/PHASE_<N>.md` — implementation sequencing and deliverables for that phase.
4. `AGENT-BEHAVIOUR/CODE_DESIGN.md` — code style and file layout.

---

## 4. Repository filesystem

### 4.1 Repo root (today)

```
/
├── AGENTS.md                    ← stub; points here
├── README.md                    ← human + agent pointer to this file
├── .gitignore
├── AGENT_FILES/                 ← all agent/human planning docs (see §4.2)
└── CHAT.md                      ← local-only; gitignored; not on GitHub
```

### 4.2 `AGENT_FILES/` — planning and instructions

```
AGENT_FILES/
├── AGENTS.md                    ← this file (start here)
├── AGENT-BEHAVIOUR/
│   ├── PROTOCOL.md              ← session behavior, RunPod paths, checklist
│   └── CODE_DESIGN.md           ← flat code layout, naming, docstrings, tooling
├── KNOWLEDGE/
│   ├── UNDERSTANDING.md         ← full architecture comprehension (§0–§14)
│   └── hierarchical_jepa_flow_architecture_brief.pdf
├── PHASES/
│   ├── PHASE_1.md               ← Stages 0+1: encoder, bottleneck, F_c, SIGReg
│   ├── PHASE_2.md               ← Stages 2+3: F_e, shuffled-c bypass test
│   └── PHASE_3.md               ← Stage 4: frame generator D + full eval
└── SETUPS/
    ├── SETUP.md                 ← human: Path A (first time) / Path B (push cycle)
    └── SETUP_POD.md             ← human: RunPod volume, SSH, troubleshooting
```

| Path | Role |
|---|---|
| `AGENTS.md` | Entry compass — project summary, filesystem, read order (this file). |
| `AGENT-BEHAVIOUR/PROTOCOL.md` | Operating procedure for every agent session. |
| `AGENT-BEHAVIOUR/CODE_DESIGN.md` | Code conventions; target layout for implementation. |
| `KNOWLEDGE/UNDERSTANDING.md` | Expanded brief: shapes, losses, EMA, stages, diagnostics, §14 decisions. |
| `KNOWLEDGE/*.pdf` | Original locked design brief. |
| `PHASES/PHASE_1.md` | Build spec: empty repo → Stage 1 training + acceptance gates. |
| `PHASES/PHASE_2.md` | Extends Phase 1; fine flow + hierarchy tests. |
| `PHASES/PHASE_3.md` | Frame gen + seven-test eval; completes v0. |
| `SETUPS/SETUP.md` | Human operator workflow (not agent implementation steps). |
| `SETUPS/SETUP_POD.md` | RunPod-specific reference and provenance links. |

### 4.3 Target code layout (after Phase 1 starts)

Implementation lives at **repo root** — flat, no `src/` package. From `CODE_DESIGN.md`:

```
config.py         # dataclass configs; §2.6 constants; path defaults
data.py           # SSv2 dataset, preprocessing, tubelet dropout
models.py         # encoder E, bottleneck B, flows F_c/F_e, generator D, EMA wrappers
losses.py         # flow-matching + SIGReg (pure tensor math, no nn.Modules)
diagnostics.py    # collapse/bypass probes → metrics dicts
train.py          # training loop, EMA update, wandb, CLI entry point
make_subset.py    # creates /workspace/data/ssv2_tiny (Phase 1 §5)
requirements.txt
pyproject.toml
```

Optional later: `eval.py` (Phase 3). **Disregard** any old Python on the RunPod volume — build fresh from phase docs.

### 4.4 RunPod volume layout (runtime, not in git)

```
/workspace/
├── hierarchal-jepa-flow-world-model/   ← this git repo
├── data/
│   ├── ssv2/                           ← full SSv2 symlinks + labels.json
│   └── ssv2_tiny/                      ← ~4k/256 subset (created in Phase 1)
├── ssv2_raw/                           ← raw .webm (read-only)
├── checkpoints/                        ← training checkpoints
└── hf_cache/                           ← Hugging Face cache (Phase 3 VAE)
```

Override data root locally with env var `JEPA_DATA_ROOT` only — no auto-detection.

---

## 5. Phases and training stages (orientation)

| Agent phase | Brief stages | Step budget (150k total) | Delivers |
|---|---|---|---|
| Phase 1 | 0 + 1 | 30k + 25k | Coarse hierarchy: E, B, EMA, F_c, SIGReg |
| Phase 2 | 2 + 3 | 50k | Fine flow F_e; shuffled-c is the central contract |
| Phase 3 | 4 + eval | 45k | Frame generator D; seven diagnostic tests |

One phase per session goal. Do not start Phase 2 until Phase 1 acceptance gates pass (human confirms).

---

## 6. Locked facts agents get wrong without reading docs

- **Fresh codebase** — reuse SSv2 data on the volume, not legacy code/checkpoints.
- **Stop-gradient** — centralize via `as_target()`; see `PROTOCOL.md` §7 and `UNDERSTANDING.md` §6.
- **Shape contracts** — every public tensor function documents shapes from `UNDERSTANDING.md` §2.
- **Constants** — dims, LRs, lambdas, thresholds only from `UNDERSTANDING.md` §2.6.
- **Bypass tests** — architecture exists to pass them; do not mark components done without diagnostics.
- **Human escalation** — ask before changing §2.6 constants, file layout, or gradient routing ambiguity.

---

## 7. Session checklist (copy from PROTOCOL)

```
[ ] Read AGENT_FILES/AGENTS.md (this file) if first clone.
[ ] Read AGENT-BEHAVIOUR/PROTOCOL.md.
[ ] Read AGENT-BEHAVIOUR/CODE_DESIGN.md.
[ ] Read KNOWLEDGE/UNDERSTANDING.md §0–§3 and §2.6; skim brief sections cited by phase doc.
[ ] Human says "execute Phase N" → open PHASES/PHASE_N.md; follow Workflow in order.
[ ] Run verification after each deliverable; run Acceptance Gate at phase end; report metrics.
[ ] Do not begin next phase until human confirms acceptance.
```

---

## 8. Typical agent prompt

> Read `AGENT_FILES/AGENTS.md`, then `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` and `CODE_DESIGN.md`. Execute Phase 1 per `AGENT_FILES/PHASES/PHASE_1.md`.

Adjust phase number when Phase 1+ is complete.
