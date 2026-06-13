# 04 — Execution phases (run-gated roadmap)

> **What this file is.** The *ordering* of the work as phases, where **each
> phase ends with a pipeline run** whose result gates the next phase. This is
> the map; [`TASKS.md`](TASKS.md) is the step-by-step (the `A4.x` tasks this
> file references). Read [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) and
> [`ANALYSIS_AND_DECISIONS.md`](ANALYSIS_AND_DECISIONS.md) first.
>
> **Principle.** Change as little as possible per run so each result is
> attributable. Cheap runs (short, `ssv2_tiny`) come before expensive ones
> (full SSv2, hours). Every code change is flag-gated (default off) so the
> v0.2 baseline is always reproducible.

---

## The runs at a glance

| Phase | Code work | Run that gates it | Cost | Question it answers |
|---|---|---|---|---|
| **P1** | A4.0 instrument + A4.1 init | Run #1: short, tiny, init ON / reg OFF | ~cheap (~30 min) | Which collapse (slot vs feature)? Did init sharpen attention? Does instrumentation work? |
| **P2** | A4.2 VICReg-C (built during P1's run) | Run #2: larger/full data, the A/B | expensive (hours) | Does decorrelation lift rank **and** hold copy-ratio? |
| **P3** | A4.3 SIGReg / raise λ_cov / task-difficulty (conditional) | Run #3: another A/B | expensive | Only if P2 is partial — which escalation works? |
| **P4** | lock winning flags as default | Run #4: full Phase-1 acceptance run | expensive | Do we clear the Phase 1 gates and proceed to project Phase 2? |

---

## Phase P1 — Instrument & init, then a short diagnostic run

**Goal:** make the failure *visible* and take the cheap init shot, before
spending money.

**Code (agent, local — no GPU):**
- A4.0 — add `attention_entropy` and `slot_diversity_rank` (split slot
  redundancy from feature correlation).
- A4.1 — init flags (`zero_init_out_mlp`, `query_init`), default off.
- *Concurrently begin A4.2 (VICReg-C) — it doesn't need to be done to launch
  Run #1.*

**Local gate (before any run):** `smoke_test_models`, `smoke_test_diagnostics`,
and `python train.py --stage0-only` all pass; with all flags **off** the loss
and grads are byte-identical to today's baseline.

**Run #1 (human, H4.2):** ~300–500 steps on `ssv2_tiny`, **init flags ON,
reg OFF**, `--log-every 50 --diag-every 100`.

**Decision gate:**
- Instrumentation prints sane numbers → continue.
- Read `attention_entropy` at step 0 (did init sharpen it vs baseline?),
  `slot_diversity_rank` vs cross-video `c_effective_rank` (which collapse
  dominates?), and the early rank trend.
- **Expectation:** init is hygiene — it may sharpen attention and nudge the
  *starting* rank, but it is unlikely to fix a training-dynamics collapse.
  This run is **diagnostic, not a verdict.** Proceed to P2 regardless; carry
  the init flags forward as "on" (they're harmless and verified).
- *Note:* 300–500 steps won't show the full rank trajectory (Run 1's plateau
  took thousands of steps). Don't over-read a short run; its job is to
  de-risk the instrumentation and reveal *which* collapse, not to conclude.

---

## Phase P2 — VICReg-C, the main A/B (the real experiment)

**Goal:** test the primary hypothesis — a decorrelation penalty lifts rank
without breaking prediction.

**Code (agent):** A4.2 complete and verified (λ_cov = 0 reproduces baseline).

**Pick λ_cov (calibration, not a guess):** first log `L_cov` at λ_cov = 0 to
see its raw magnitude, then set λ_cov so that `λ_cov · L_cov` is ~1–10% of
`L_flow` at the start. Sweep around that (e.g. ×0.3 and ×3) only if needed.

**Decide the run matrix (human + tech lead, per `ANALYSIS_AND_DECISIONS.md` §3):**
- **2-run** (full attribution): Run A = no reg, full data; Run B = VICReg-C,
  full data. Separates the data effect from the regularizer effect.
- **1-run** (cheaper): Run B only; ship if rank **and** copy-ratio both improve.

**Run #2 (human, H4.3):** larger/full SSv2, init flags ON, λ_cov set as above.
Report `c_effective_rank` **and** `coarse_vs_copy_ratio` *together* at
milestones, plus the standard guards.

**Decision gate (success criteria, `DETAILED_UNDERSTAND.md` §7):**
- **Success** = rank > 30 (toward 60) **AND** copy-ratio improves/holds →
  **go to P4.**
- **Rank flat near ~5** = the term isn't biting → **go to P3** (raise λ_cov or
  SIGReg).
- **Rank up but copy-ratio down** = Goodhart → reduce λ_cov and re-run; if
  irreconcilable, **go to P3** (task-difficulty path with tech lead).

---

## Phase P3 — Escalation (only if P2 is partial)

**Goal:** find the lever P2 didn't provide. Pick **one** path based on P2's
failure shape:

- **Rank wouldn't move** → A4.3 SIGReg on `c_t` only (stronger,
  distribution-level), or a higher λ_cov. (SIGReg and λ_cov are mutually
  exclusive in one run — don't confound two decorrelation terms.)
- **Goodhart (rank↑, quality↓)** → the regularizer is fighting the objective;
  revisit **task difficulty** (frame-overlap / predict-k-tubelets-ahead,
  `DETAILED_UNDERSTAND.md` §6) with the tech lead — this is the Phase-4-flavored
  lever, pulled early only if forced.

**Run #3:** another A/B with the chosen escalation. Same success gate as P2.
Loop back here at most once or twice before escalating the *decision* to the
tech lead rather than burning more runs.

---

## Phase P4 — Lock the fix & run the Phase-1 acceptance run

**Goal:** rejoin the main project pipeline.

**Code (agent):** set the winning flags (init + λ_cov, or SIGReg) as the
Phase 1 defaults in `config.py`. Flag the **supervisor sign-off**: a kept
covariance/SIGReg term reverses the v0.2 "no covariance loss initially"
directive (`DETAILED_UNDERSTAND.md` §8) — this needs explicit approval before
it becomes the standing default.

**Run #4 (human):** a full Phase-1 acceptance run — this is effectively a fresh
[`../02-LAUNCH-FULL-PHASE-1-RUN/`](../02-LAUNCH-FULL-PHASE-1-RUN/) launch with
the new defaults, judged against the real Phase 1 gates
([`../../PHASES/PHASE_1.md`](../../PHASES/PHASE_1.md) §12: `coarse_vs_copy_ratio`
≤ 0.70, etc.).

**Decision gate:** Phase 1 gates pass → **proceed to project Phase 2.** This
closes Plan Phase 04 (A4.5) and hands back to the `02` flow.

---

## Quick reference: which flags are on in which run

| | init flags | λ_cov | SIGReg | dataset |
|---|---|---|---|---|
| Baseline (Run 1, have it) | off | 0 | off | tiny |
| Run #1 (P1) | **on** | 0 | off | tiny (short) |
| Run #2 (P2) | on | **set** | off | full (A/B) |
| Run #3 (P3) | on | set *or* 0 | **on** *(if escalating)* | full (A/B) |
| Run #4 (P4) | locked | locked | locked | full (acceptance) |
