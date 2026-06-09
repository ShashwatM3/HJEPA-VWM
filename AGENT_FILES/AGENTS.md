# AGENTS.md — Coding agent entry point

> **Audience:** Any AI coding agent that clones or opens this repository.
> **Purpose:** Replace generic repo guessing with a focused map of the project, the filesystem, and the documents you must read before writing code.

---

## 1. What this repository is

**HJEPA-VWM** (Hierarchical JEPA-Flow Video World Model, v0.2) is a **video world model** that learns a **two-level latent hierarchy**:

- **Abstract latent `c_t`** — future-relevant structure (32 tokens × 256 dims), from a **trainable bottleneck**.
- **Detailed latent `e_t`** — texture and local visual detail (**1024 context tokens × 1024 dims**), from a **frozen pretrained V-JEPA 2 ViT-L/16 encoder**.

The model predicts **future latents**, not pixels directly. Two **flow-matching** networks (`F_c`, `F_e`) learn velocity fields from noise to future targets `c⁺_{t+k} = B_EMA(E(x_{≤t+k}))`. The target comes from an **EMA bottleneck branch** (always stop-gradient; the encoder is frozen and shared). Collapse prevention is a **variance floor on `c_t`** (no SIGReg/VICReg). A **frame generator `D`** (Stage 4 only, after latent hierarchy is verified) renders pixels in frozen VAE latent space.

> **v0.2 update:** the encoder is now **pretrained + frozen** (was trained from scratch), collapse
> prevention is the **variance floor** (was SIGReg), and **multi-horizon prediction** is a new
> **Phase 4** (deferred). Read [`KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md`](KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md)
> and [`KNOWLEDGE/BRIEF_V0_2.md`](KNOWLEDGE/BRIEF_V0_2.md) first.

This is **not** a video diffusion model. The compressed predictive state is the point; bypass tests prove the hierarchy is real.

**Current repo state (planning phase):** There is **no Python implementation yet**. The repository contains agent instructions, architecture comprehension, phased build specs, and human RunPod setup guides. Implementation is produced by an agent executing `PHASES/PHASE_1.md` → `PHASE_2.md` → `PHASE_3.md` after a human completes `SETUPS/SETUP.md`.

**Training target:** RunPod GPU pod with a network volume. Dataset: Something-Something V2 (SSv2). Code is edited locally → git push → SSH → `git pull` → train on pod.

---

## 2. Mandatory read order (do this before any code)

| Order | File | Why |
|---|---|---|
| **1** | [`AGENT-BEHAVIOUR/PROTOCOL.md`](AGENT-BEHAVIOUR/PROTOCOL.md) | **How you must behave** — phase execution, escalation, stop-gradient rules, acceptance gates. Read at the **start of every session**. |
| **1b** | [`SETUPS/VOLUME_LAYOUT.md`](SETUPS/VOLUME_LAYOUT.md) | **RunPod volume** — what exists on the network volume today vs target layout; path contract for `config.py`. Read before touching `data.py`, `config.py`, or `make_subset.py`. |
| **2** | [`AGENT-BEHAVIOUR/CODE_DESIGN.md`](AGENT-BEHAVIOUR/CODE_DESIGN.md) | **How you must write code** — flat 5–6 file layout, naming map, docstrings, `as_target()` pattern, black/ruff. Read before creating or renaming any file. |
| **2b** | [`KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md`](KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md) | **The v0.2 update explained** — frozen encoder, flow target, variance floor, multi-horizon. Read before the brief. |
| **3** | [`KNOWLEDGE/UNDERSTANDING.md`](KNOWLEDGE/UNDERSTANDING.md) §0–§3, §2.6, §6 | **What you are building** — shapes, locked constants, modules, stop-gradient table. Re-read §2, §2.6, §6 whenever touching latents or losses. |
| **4** | [`KNOWLEDGE/BRIEF_V0_2.md`](KNOWLEDGE/BRIEF_V0_2.md) | **Authoritative working spec** (v0.2). The original PDF is preserved verbatim as [`BRIEF_V0_1.md`](KNOWLEDGE/BRIEF_V0_1.md). For the encoder choice see [`FROZEN_ENCODER_RESEARCH.md`](KNOWLEDGE/FROZEN_ENCODER_RESEARCH.md). |
| **5** | [`PHASES/PHASE_<N>.md`](PHASES/PHASE_1.md) | **What to build this session** — only after the human says "execute Phase N." Follow the workflow table top-to-bottom. |

**Do not read** generic tutorials or substitute a different architecture. **Do not start coding** until steps 1–2 are done and step 5 is opened.

Human operators (not agents) use [`SETUPS/SETUP.md`](SETUPS/SETUP.md) and [`SETUPS/SETUP_POD.md`](SETUPS/SETUP_POD.md) for first-time RunPod setup and troubleshooting.

---

## 3. Document precedence (when sources conflict)

1. `KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md` — the v0.2 update; **top authority** on architecture intent.
2. `KNOWLEDGE/BRIEF_V0_2.md` — the edited working brief.
3. `KNOWLEDGE/UNDERSTANDING.md` §2.6 — numerical constants.
4. Active `PHASES/PHASE_<N>.md` — implementation sequencing and deliverables for that phase.
5. `KNOWLEDGE/BRIEF_V0_1.md` / the original PDF — historical baseline; superseded where it conflicts.
6. `AGENT-BEHAVIOUR/CODE_DESIGN.md` — code style and file layout.

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
│   ├── UNDERSTANDING.md         ← full architecture comprehension (§0–§14), v0.2
│   ├── SUPERVISOR_FEEDBACK_EXPLAINED.md  ← v0.2 update, taught from first principles
│   ├── BRIEF_V0_1.md            ← exact replica of the original PDF (frozen baseline)
│   ├── BRIEF_V0_2.md            ← working brief (PDF + v0.2 edits)
│   ├── FROZEN_ENCODER_RESEARCH.md        ← encoder survey + recommendation + cascade
│   ├── ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md  ← roadmap + per-file change index
│   └── hierarchical_jepa_flow_architecture_brief.pdf  ← original (see BRIEF_V0_1.md)
├── PHASES/
│   ├── PHASE_1.md               ← Stages 0+1: frozen encoder, bottleneck, F_c, variance floor
│   ├── PHASE_2.md               ← Stages 2+3: F_e, shuffled-c bypass test
│   ├── PHASE_3.md               ← Stage 4: frame generator D + full eval
│   └── PHASE_4.md               ← Multi-horizon prediction (NEW; deferred)
└── SETUPS/
    ├── VOLUME_LAYOUT.md         ← current vs target network volume; path contract (agents + humans)
    ├── SETUP.md                 ← human: Path A (first time) / Path B (push cycle)
    └── SETUP_POD.md             ← human: SSH, troubleshooting (points to VOLUME_LAYOUT.md)
```

| Path | Role |
|---|---|
| `AGENTS.md` | Entry compass — project summary, filesystem, read order (this file). |
| `AGENT-BEHAVIOUR/PROTOCOL.md` | Operating procedure for every agent session. |
| `AGENT-BEHAVIOUR/CODE_DESIGN.md` | Code conventions; target layout for implementation. |
| `KNOWLEDGE/UNDERSTANDING.md` | Expanded brief: shapes, losses, EMA, stages, diagnostics, §14 decisions (v0.2). |
| `KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md` | The v0.2 update taught from first principles. |
| `KNOWLEDGE/BRIEF_V0_1.md` / `BRIEF_V0_2.md` | Exact PDF replica (baseline) / edited working brief. |
| `KNOWLEDGE/FROZEN_ENCODER_RESEARCH.md` | Encoder survey, recommendation, architectural cascade. |
| `KNOWLEDGE/ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md` | Roadmap + per-file change index. |
| `KNOWLEDGE/*.pdf` | Original locked design brief (transcribed in `BRIEF_V0_1.md`). |
| `PHASES/PHASE_1.md` | Build spec: empty repo → Stage 1 training + acceptance gates. |
| `PHASES/PHASE_2.md` | Extends Phase 1; fine flow + hierarchy tests. |
| `PHASES/PHASE_3.md` | Frame gen + seven-test eval; completes v0. |
| `PHASES/PHASE_4.md` | Multi-horizon prediction (NEW; deferred). |
| `SETUPS/VOLUME_LAYOUT.md` | **Canonical** RunPod volume tree: current state, target state, `config.py` paths. |
| `SETUPS/SETUP.md` | Human operator workflow (not agent implementation steps). |
| `SETUPS/SETUP_POD.md` | RunPod SSH and troubleshooting (volume structure → `VOLUME_LAYOUT.md`). |

### 4.3 Target code layout (after Phase 1 starts)

Implementation lives at **repo root** — flat, no `src/` package. From `CODE_DESIGN.md`:

```
config.py         # dataclass configs; §2.6 constants; path defaults
data.py           # SSv2 dataset, preprocessing (no tubelet dropout in v0.2)
models.py         # frozen encoder E, bottleneck B, flows F_c/F_e, generator D, B_EMA wrapper
losses.py         # flow-matching + variance floor (pure tensor math, no nn.Modules)
diagnostics.py    # collapse/bypass probes → metrics dicts
train.py          # training loop, EMA update, wandb, CLI entry point
make_subset.py    # creates /workspace/data/ssv2_tiny (Phase 1 §5)
requirements.txt
pyproject.toml
```

Optional later: `eval.py` (Phase 3). **Disregard** any old Python on the RunPod volume — build fresh from phase docs.

### 4.4 RunPod network volume (runtime, not in git)

**Full reference:** [`SETUPS/VOLUME_LAYOUT.md`](SETUPS/VOLUME_LAYOUT.md) — current volume contents (SSv2 already on disk; `ssv2_tiny` not created yet), target layout after migration, and path defaults for `config.py`.

Summary: code lives in `/workspace/hierarchal-jepa-flow-world-model/`; `data/`, `checkpoints/`, `ssv2_raw/`, and `hf_cache/` are **siblings** under `/workspace/`. Override data root locally with env var `JEPA_DATA_ROOT` only — no auto-detection.

---

## 5. Phases and training stages (orientation)

| Agent phase | Brief stages | Step budget (150k total) | Delivers |
|---|---|---|---|
| Phase 1 | 0 + 1 | 30k + 25k | Coarse hierarchy: **frozen E**, B, **B_EMA**, F_c, **variance floor** + 3 metrics |
| Phase 2 | 2 + 3 | 50k | Fine flow F_e; shuffled-c is the central contract |
| Phase 3 | 4 + eval | 45k | Frame generator D; seven diagnostic tests |
| Phase 4 | multi-horizon | (deferred) | `k∈{4,8,16,32}` + horizon embedding `h_k`; one shared predictor |

One phase per session goal. Do not start the next phase until the current one's acceptance gates pass (human confirms). **Phase 4 is deferred** — do not start until Phases 1–3 are healthy and the human says so.

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
[ ] Read SETUPS/VOLUME_LAYOUT.md (before data paths or config).
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
