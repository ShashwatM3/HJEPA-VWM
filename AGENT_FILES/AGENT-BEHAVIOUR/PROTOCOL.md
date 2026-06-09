# PROTOCOL.md — Agent Operating Procedure

> **Audience:** AI coding agents implementing the Hierarchical JEPA-Flow video world model (v0).
> **Read [`AGENT_FILES/AGENTS.md`](../AGENTS.md) on first clone, then this file at the start of every session before touching code.**

---

## 0. Repository layout (agent docs)

Entry point for new agents: **[`AGENT_FILES/AGENTS.md`](../AGENTS.md)** (project summary, filesystem map, read order).

All planning and agent instruction files live under **`AGENT_FILES/`** in this repository:

```
AGENT_FILES/
├── AGENTS.md                    ← entry compass (read first on clone)
├── AGENT-BEHAVIOUR/
│   ├── PROTOCOL.md              ← this file
│   └── CODE_DESIGN.md
├── KNOWLEDGE/
│   ├── UNDERSTANDING.md
│   ├── SUPERVISOR_FEEDBACK_EXPLAINED.md
│   ├── BRIEF_V0_1.md            ← exact PDF replica (baseline)
│   ├── BRIEF_V0_2.md            ← working brief (v0.2)
│   ├── FROZEN_ENCODER_RESEARCH.md
│   ├── ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md
│   └── hierarchical_jepa_flow_architecture_brief.pdf
├── PHASES/
│   ├── PHASE_1.md
│   ├── PHASE_2.md
│   ├── PHASE_3.md
│   └── PHASE_4.md               ← multi-horizon (deferred)
└── SETUPS/
    ├── SETUP.md
    └── SETUP_POD.md
```

When any doc names another file, use the **full path from repo root** (e.g. `AGENT_FILES/PHASES/PHASE_1.md`).

---

## 1. Document map — what each file is for

| Path | Role | When to read |
|---|---|---|
| `AGENT_FILES/AGENTS.md` | **Entry compass** — project summary, filesystem map, mandatory read order. | First clone / first session; when unsure where anything lives. |
| `AGENT_FILES/KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md` | **The v0.2 update** — frozen encoder, flow target, variance floor, multi-horizon — taught from first principles. **Top authority on intent.** | First, when orienting to the current (v0.2) architecture. |
| `AGENT_FILES/KNOWLEDGE/BRIEF_V0_2.md` | **Authoritative working spec** (PDF + v0.2 edits). Original PDF preserved as `BRIEF_V0_1.md`. | When a phase doc or UNDERSTANDING.md references a brief section; when verifying a design constraint. |
| `AGENT_FILES/KNOWLEDGE/FROZEN_ENCODER_RESEARCH.md` | Encoder choice (V-JEPA 2 ViT-L, `D_e=1024`), specs, and the architectural cascade. | Before touching the encoder, `D_e`, token counts, or normalization. |
| `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` | **Expanded comprehension reference** — shape contracts (§2), locked constants (§2.6), modules (§3), forward pass (§4), losses (§5), stop-gradient (§6), EMA (§7), training schedule (§8), diagnostics (§9). | Before writing any function that touches latents, losses, gradients, or training stages. Re-read §2, §2.6, and §6 on every session. |
| `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` | **Code style and layout** — flat 5–6 file repo, naming map, docstring template, stop-gradient helper pattern, tooling (black, ruff). | Before creating or renaming any file; before writing docstrings. |
| `AGENT_FILES/PHASES/PHASE_1.md` | **Implementation spec for Phase 1** — Coarse Hierarchy (Stages 0 + 1). Complete build instructions from empty repo to a training run that passes Stage 1 gates. | When the human says "execute Phase 1." |
| `AGENT_FILES/PHASES/PHASE_2.md` | **Implementation spec for Phase 2** — Detailed Hierarchy (Stages 2 + 3). Extends Phase 1 codebase; no rework of Phase 1 modules. | When the human says "execute Phase 2." Requires Phase 1 complete. |
| `AGENT_FILES/PHASES/PHASE_3.md` | **Implementation spec for Phase 3** — Frame Generation (Stage 4) + inference utilities. Completes v0. | When the human says "execute Phase 3." Requires Phase 2 complete. |
| `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` (this file) | **How you operate** — behavior rules, research permission, escalation triggers. | Every session, first. |
| `AGENT_FILES/SETUPS/SETUP.md` | **Human operator guide** — Path A (first time) and Path B (local changes → train → metrics). Read before first RunPod run. | **Human operator:** before first training run and after each local push cycle. |
| `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md` | **RunPod volume structure** — current vs target layout, path contract. | Before `config.py` / `data.py` / `make_subset.py`; when verifying data on pod. |
| `AGENT_FILES/SETUPS/SETUP_POD.md` | **RunPod reference** — SSH, troubleshooting. | When debugging pod/volume/SSH issues. |

**Precedence when documents conflict:**
1. `AGENT_FILES/KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md` + `BRIEF_V0_2.md` win on architecture intent (v0.2). `BRIEF_V0_1.md` / the PDF is the superseded baseline.
2. `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2.6 wins on numerical constants.
3. The active `AGENT_FILES/PHASES/PHASE_<N>.md` wins on implementation sequencing and deliverables for that phase.
4. `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` wins on code style and file layout.

---

## 2. Operating mode

1. **Identify the active phase.** The human will say "execute Phase N." Open `AGENT_FILES/PHASES/PHASE_N.md` and follow its workflow section top-to-bottom, in order. Do not skip steps. Do not reorder steps unless the phase doc explicitly allows it.
2. **Treat the phase doc as the session source of truth** for what to build, in what order, and what "done" means. Deviate only after surfacing the deviation to the human and receiving approval.
3. **One phase per session goal.** Do not start Phase 2 work while Phase 1 acceptance gates are unmet. Do not start Phase 3 while Phase 2 gates are unmet.
4. **No scaffolding-only deliverables.** Every phase ends with a runnable training command on RunPod that produces measurable output (loss curves, diagnostic metrics, checkpoints). If you cannot run it, the phase is not done.
5. **Fresh codebase.** Disregard any prior Python implementation on the RunPod volume. Build from the phase specs. The only pre-existing assets to reuse are **data on the network volume** (SSv2 symlinks, raw `.webm` files) — not old code.

---

## 3. Deployment target — RunPod

**Canonical volume reference:** [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../SETUPS/VOLUME_LAYOUT.md) — current volume state (SSv2 on disk; `ssv2_tiny` not yet created), target layout after migration, path table for `config.py`, and who runs `make_subset.py`.

All code is written for the **target** layout: repo at `/workspace/hierarchal-jepa-flow-world-model/`; `data/`, `checkpoints/`, `ssv2_raw/`, and `hf_cache/` as siblings under `/workspace/`. Override data root locally with env var `JEPA_DATA_ROOT` only — no auto-detection.

**Workflow:** local dev → git push → SSH into RunPod (see `AGENT_FILES/SETUPS/SETUP.md` Path B) → `git pull` → run. First-time setup: `AGENT_FILES/SETUPS/SETUP.md` Path A.

---

## 4. Research permission

You **may** search papers, official docs, and reference implementations when:

- A phase doc says "refer to external source" (e.g., V-JEPA 2 `from_pretrained` usage, adaLN-Zero, diffusers VAE API).
- An API signature or library behavior is ambiguous (e.g., `diffusers.AutoencoderKL` encode/decode contract).
- You need to verify a formula (flow matching interpolation, EMA update rule).

**How to use research results:**
- Lock the finding into code comments referencing the source (paper name, URL, or repo path).
- If research contradicts `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2.6 or the brief, **stop and ask the human** — do not silently override.
- Prefer primary sources: brief → `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` → paper → reference impl.

You **must not** substitute a different architecture, loss, or training schedule because a blog post suggested it. v0 follows the brief.

---

## 5. Ask-the-human protocol

**Must ask before proceeding** (do not guess):

- Any change to a non-negotiable constraint (brief §11 / `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §10).
- Any change to locked constants in §2.6 (dims, LRs, step counts, thresholds).
- Any deviation from the flat file layout in `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`.
- Ambiguity in a phase doc where two interpretations would produce different gradient paths.
- Missing file on the network volume that blocks execution (broken symlinks, absent labels.json).

**May proceed with a documented assumption** (log it in the commit message and tell the human):

- decord worker count tuning for RunPod I/O.
- W&B project name / run name.
- Minor logging verbosity.

**Never ask the human to run commands you can run yourself** on RunPod via SSH instructions in the phase doc — but if you only have local access, produce exact commands for them.

---

## 6. Shape-contract discipline

Every function that touches tensors gets a docstring per `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §4:

1. One-line summary.
2. Plain-English "why."
3. `Args` / `Returns` with shapes using named dimensions from the brief (`B`, `N_ctx`, `D_e`, `N_c`, `D_c`, etc.).

**Refuse to write** a public function without a shape contract. Cross-check every shape against `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2 and §2.6.

Optional runtime asserts behind `cfg.debug_shapes` — default `True` during Phase 1, flip to `False` only when the human requests production speed.

---

## 7. Stop-gradient hygiene

Centralize detaches in one helper (`as_target()` per `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §6). The complete rules:

| Tensor | Stop-grad when | Rationale |
|---|---|---|
| `e_plus`, `c_plus` | Always | EMA target branch outputs |
| `c_hat` fed into `F_e` | Always in Stage 3+ | Prevents L_e from corrupting F_c |
| `e_hat` fed into `D` | Always in Stage 4 | Frame gen must not rewrite world model |
| `E` (frozen encoder) | Never receives backprop | Pretrained, frozen; `requires_grad=False` both branches |
| `B_EMA` | Never receives backprop | EMA update only (bottleneck) |
| `e_t`, `c_t` on conditioning path | No stop-grad during latent training | Encoder must learn |

Comment each `as_target()` call with the failure mode it prevents.

---

## 8. Bypass-test mindset

The architecture exists to pass the hierarchy bypass tests (brief §9). Rules:

- Do not mark a component "done" until its diagnostic exists and runs in the training loop or `eval.py`.
- Phase 1: F_c vs copy/batch-mean baselines, latent std, effective rank, gradient health.
- Phase 2: shuffled-c test, zero-c ablation, teacher-vs-predicted gap — **the shuffled-c test is the central contract.**
- Phase 3: decoder dependency (shuffled `e_hat`), full seven-test eval script.

Thresholds are locked in `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2.6 (Diagnostic thresholds).

---

## 9. How to execute a phase — session checklist

```
[ ] 1. Read AGENT_FILES/AGENTS.md (if first session).
[ ] 2. Read AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md (this file).
[ ] 3. Read AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md.
[ ] 4. Read AGENT_FILES/KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md; skim BRIEF_V0_2.md sections cited by the phase doc.
[ ] 5. Read AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md §0–§3 and §2.6 in full.
[ ] 6. Open AGENT_FILES/PHASES/PHASE_<N>.md; read the Workflow section; execute steps in order.
[ ] 7. After each major deliverable (file or milestone), run the phase doc's verification command.
[ ] 8. At phase end, run the Acceptance Gate section verbatim; report metrics to the human.
[ ] 9. Do not begin the next phase until the human confirms acceptance.
```

---

## 10. Tooling and quality bar

From `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §8 — enforce from Phase 1:

- `black` (line length 100)
- `ruff`
- type hints on all public signatures
- `requirements.txt` with pinned major versions
- optional `.pre-commit-config.yaml` mirroring black + ruff

Log to Weights & Biases (`wandb`) from Phase 1 onward: loss components, LR, EMA momentum, diagnostic metrics every 500–1000 steps.

---

## 11. What "phase complete" means

A phase is complete when **all** of the following hold:

1. Every file listed in the phase doc's Deliverables section exists and matches the spec.
2. The phase doc's Verification commands exit 0 on RunPod (or the human confirms they ran successfully).
3. Every Acceptance Gate threshold in the phase doc is met or explicitly flagged to the human with numbers.
4. README section for that phase is written (what works, how to run, expected runtime).
5. No TODO/FIXME left in code paths on the hot training loop unless the human approved deferral.

Report completion to the human with: commands run, final metrics, checkpoint path, and anything that failed a soft threshold.
