# ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md — Migration roadmap & per-file change index

> **What this file is.** The coordination document for the supervisor's update. It states the
> incremental rollout policy, the revised phase map, the document-precedence order, and a precise
> per-file list of *what changes and why*. Read this after
> [`SUPERVISOR_FEEDBACK_EXPLAINED.md`](SUPERVISOR_FEEDBACK_EXPLAINED.md) and
> [`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md); it ties them to concrete edits.
>
> **Status.** The pick-independent docs (this file, the feedback explainer, the encoder research, and
> `BRIEF_V0_1.md`) are **done**. The numeric in-place rewrites (`BRIEF_V0_2.md`, `UNDERSTANDING.md`,
> `PHASE_1.md`, `PHASE_2.md`, `PHASE_3.md`, new `PHASE_4.md`, and the behaviour/index docs) are
> **pending the human's encoder + resolution decision** (`FROZEN_ENCODER_RESEARCH.md` §8).

---

## 1. The rollout policy: one change at a time

The supervisor's update bundles **two** large changes — (1) a frozen pretrained encoder and (2)
multi-horizon prediction. **We deliberately do not land both at once.** Landing them together would
make any failure ambiguous (encoder swap vs horizon generalization) and would rewrite half the
codebase in one unverifiable jump.

**Order of operations:**

1. **Immediate next step — frozen-encoder swap.** Replace the from-scratch encoder (online + target)
   with a single frozen pretrained ViT; collapse EMA to the bottleneck only; replace SIGReg with the
   variance floor on `c_t`. (Research first, then code — see scope note below.)
2. **Second step — new metrics.** Add the three required monitors to W&B: variance of `c_t`,
   cross-video cosine similarity of `c_t`, effective rank of `c_t`.
3. **Deferred — multi-horizon prediction.** Becomes the new **Phase 4**. Not started until the
   existing phases are healthy with the new encoder.

> **Scope note (current task).** The present work produces **documentation/Markdown only** — research
> and updated specs. **No Python is written yet** (none exists). "Change the files" here means update
> the knowledge/brief/phase docs, not implement code.

---

## 2. Revised phase map

| Phase | Name | What it delivers | Change from before |
|---|---|---|---|
| **1** | Coarse hierarchy | Frozen pretrained ViT `E` → `e_t`; trainable bottleneck `B` → `c_t`; coarse flow `F_c`; **variance floor** on `c_t`; EMA on `B` only; **the 3 new metrics**. | Encoder is now frozen/pretrained; SIGReg → variance floor; EMA scope shrinks; new metrics added. |
| **2** | Detailed hierarchy | Fine flow `F_e` (teacher-forced → predicted-coarse); shuffled-c bypass test. | Consumes the new (wider) `e_t`; otherwise structurally unchanged. |
| **3** | Frame generation | Frozen VAE + frame generator `D`; seven-test eval. | Consumes the new `e_hat`; otherwise unchanged. |
| **4** | **Multi-horizon prediction (NEW)** | `k ∈ {4,8,16,32}`, sampling `{0.30,0.30,0.25,0.15}`, learned horizon embedding `h_k` in the shared predictor. | Brand-new phase; deferred. |

Phase 5 ("optional polish") from `BRIEF_V0_1.md` §6 remains out of scope for v0.

> **Note on terminology.** "Phase" = an implementation milestone (our docs). "Stage" = a training
> stage from the brief (0–5). The brief's stages still map onto phases as before; Phase 4 adds a new
> multi-horizon training regime layered on the Stage-1 coarse objective.

---

## 3. Document precedence (updated)

When sources conflict, this is the order of authority:

1. **`SUPERVISOR_FEEDBACK_EXPLAINED.md`** (the supervisor's update) — **new top authority** on
   architecture intent for everything it touches (encoder, flow target, EMA scope, collapse loss,
   horizons, metrics).
2. **`BRIEF_V0_2.md`** — the edited brief, once produced; the working spec.
3. **`UNDERSTANDING.md` §2.6** — numerical constants (single source of truth for values), once
   updated.
4. Active **`PHASES/PHASE_<N>.md`** — implementation sequencing/deliverables for that phase.
5. **`BRIEF_V0_1.md`** / the original PDF — the historical `v0.1` baseline; superseded where it
   conflicts with the above. **Kept for reference/diff only.**
6. **`CODE_DESIGN.md`** — code style and file layout.

(The old `AGENTS.md` precedence list — PDF first — is updated accordingly; see §4.)

---

## 4. Per-file change index (what each edit will do)

> All edits below are **pending the encoder + resolution decision** so the numbers land once,
> correctly. `D_e ∈ {768, 1024}` and `N_ctx ∈ {128, 512, ...}` are written as the chosen values at
> lock time.

### `KNOWLEDGE/BRIEF_V0_2.md` — (new; duplicate of `BRIEF_V0_1.md`, then edited in place)
- §1 data path: `E` is frozen pretrained; target = `B_EMA(E(x_{≤t+k}))`; EMA on `B` only.
- §2 dimensions table: `E` row → frozen pretrained ViT (dim `D_e`); `e_t` token count/dim updated.
- §3 EMA: only `B`/`B_EMA`; encoder frozen (no `E_bar`).
- §4.3: replace SIGReg subsection with the **variance floor** (`L_var`, weight 0.1); remove
  SIGReg(e)/SIGReg(c); state "no VICReg/covariance."
- §4 total loss: coarse stage `L = L_flow + 0.1·L_var`.
- §6 schedule: Stage 1 train set = `B`, `F_c` (encoder frozen); EMA on `B` only.
- §7 optimizer: remove encoder LR; keep bottleneck/flow LRs.
- New subsection (or §10 note): **multi-horizon → Phase 4** (documented, deferred).
- §9 tests: add cross-video cosine of `c_t`; keep variance + effective rank of `c_t`.
- Add a short "Changes vs v0.1" banner at top.

### `KNOWLEDGE/UNDERSTANDING.md` — (edited in place, same style/thoroughness)
- §1 thesis: encoder frozen/pretrained; predict clip-level `c⁺_{t+k}`.
- §2 / §2.6: `D_e`, `N_ctx`, target geometry, **remove encoder depth/heads/LR**, EMA scope, **remove
  SIGReg constants**, add `L_var` weight (0.1) and variance-floor definition, add horizon constants
  as Phase-4 future.
- §3.2: replace "Online encoder E — VideoViT-Small (from scratch)" with "Frozen pretrained ViT
  (`E`)"; document load API, normalization, pos-embeds (encoder's own RoPE), token geometry.
- §3.2.1 tubelet dropout: revise per the decision (likely drop/relocate — see research §8).
- §3.4 EMA target branch: `B_EMA` only; `E` shared/frozen.
- §5 losses: §5.3 SIGReg → variance floor; §5.4 total loss updated.
- §6 stop-grad table: target now `B_EMA(E(...))`; encoder frozen row added.
- §7 EMA: schedule applies to `B` only; encoder excluded.
- §9 diagnostics: add cross-video cosine; keep variance/effective-rank thresholds for `c_t`.
- §14 decisions log: add new resolved decisions (frozen encoder, encoder pick, resolution, variance
  floor, EMA scope, multi-horizon→Phase 4) and mark superseded old decisions.

### `PHASES/PHASE_1.md` — (edited in place)
- Remove from-scratch encoder build + encoder LR group; add frozen-encoder load (HF), freeze assert.
- `models.py` section: encoder is a thin frozen wrapper; bottleneck input proj `D_e → 256`.
- `losses.py` section: SIGReg → `variance_floor(c_t)`; total `L = L_flow + 0.1·L_var`.
- `diagnostics.py` section: add `cross_video_cosine(c_t)`; keep variance + effective rank; (baselines
  unchanged).
- W&B: log the three required metrics every `diag_every`.
- config block: drop encoder LR/depth/heads; add `encoder_repo`, `d_e`, `n_ctx`, `var_floor_weight`,
  resolution; EMA on bottleneck only.
- Acceptance gates: collapse gates now on `c_t` (variance/cosine/rank); remove `e_t` SIGReg gate.

### `PHASES/PHASE_2.md` & `PHASES/PHASE_3.md` — (light edits)
- Propagate `D_e`/`N_ctx` into `F_e` memory and `D` cross-attention dims and the shape contracts.
- Confirm freeze semantics already align (encoder was always frozen in later stages; now frozen from
  the start).
- Fix cross-references to the new docs and the variance-floor loss name.

### `PHASES/PHASE_4.md` — (NEW)
- Multi-horizon spec in the existing phase-doc format: horizon sampling, `h_k` embedding wiring into
  the shared predictor, target `c⁺_{t+k}` for each `k`, loss aggregation, metrics per horizon, and
  acceptance gates. Deferred; not executed until Phases 1–3 are healthy.

### `AGENTS.md`, `AGENT-BEHAVIOUR/PROTOCOL.md`, `AGENT-BEHAVIOUR/CODE_DESIGN.md` — (touch-ups)
- `AGENTS.md`: add the four new KNOWLEDGE docs to the read order/filesystem; update precedence (§3);
  update the phase table (add Phase 4); note encoder is frozen/pretrained.
- `PROTOCOL.md`: update document map + precedence; note frozen-encoder & variance-floor rules.
- `CODE_DESIGN.md`: naming map — add `horizon_embed` (`h_k`) and clarify `E` is a frozen pretrained
  wrapper; note EMA applies to bottleneck only.

---

## 5. Invariants that do **not** change

To prevent over-correction, these remain exactly as in `BRIEF_V0_1.md`:

- The **two-level hierarchy** (`c_t` abstract, `e_t` detailed) and the coarse→fine→frame staging.
- **Rectified flow / flow matching** as the predictor mechanism (we already used it).
- **Stop-grad on all targets**; **EMA target** principle (now on `B` only).
- The **bypass-test mindset**: shuffled-c (Phase 2) and decoder-dependency (Phase 3) remain the
  central contracts.
- `c_t` **low-bandwidth** relative to `e_t` (`N_c=32, D_c=256` vs the wider frozen `e_t`).
- `L_e` must **not** backprop into `F_c` through `c_hat`; frame loss must **not** update the latent
  stack.

---

## 6. Sequencing summary

```
[DONE]    BRIEF_V0_1.md (exact replica)
[DONE]    SUPERVISOR_FEEDBACK_EXPLAINED.md
[DONE]    FROZEN_ENCODER_RESEARCH.md
[DONE]    ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md  (this file)
   │
   ▼
[HUMAN]   Decide: encoder pick + resolution + tubelet-dropout + context frames  (research §8)
   │
   ▼
[NEXT]    BRIEF_V0_2.md (duplicate v0_1 → edit in place)
[NEXT]    UNDERSTANDING.md (in place)   PHASE_1/2/3.md (in place)   PHASE_4.md (new)
[NEXT]    AGENTS.md / PROTOCOL.md / CODE_DESIGN.md (touch-ups)
   │
   ▼
[LATER]   Code (Phase 1 execution) — separate, human-triggered session
```
