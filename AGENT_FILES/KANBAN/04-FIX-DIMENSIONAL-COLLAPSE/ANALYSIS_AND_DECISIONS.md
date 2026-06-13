# 04 — Analysis & decisions: how we chose the fixes

> **Purpose.** This is the decision record. It captures (a) a graded review of
> the external opinions we collected, (b) the locked design decision with
> reasoning, and (c) the reliability/attribution logic behind the experiment
> design. Read it so you understand *why* `TASKS.md` is what it is and can
> defend or revise the choices.

---

## §1. The external review we graded (Claude "Fabel")

The tech lead forwarded the bottleneck problem list to an external model
("Claude Fabel"), which reviewed our proposed init fixes and the collapse
question. It was working **only from a screenshot** of the bottleneck's
query/attention code and **without** knowledge of the v0.2 spec history. Its
operational advice was sound; two factual claims were wrong. Grade:

| Claim | Verdict | Detail |
|---|---|---|
| Zero-init the last residual linear (bottleneck `out_mlp[-1]`) so the block starts as identity | ✅ Correct, take as-is | Valid and not yet done. `AdaLNBlock.mod` is already zero-inited; the bottleneck `out_mlp` is not. |
| The "queries point the same direction → rank 1" mechanism is backwards | ✅ Correct | Random Gaussian vectors in 256-D are near-orthogonal (concentration of measure). Direction was never the issue. |
| The real issue is **scale**: tiny queries → uniform softmax → all slots read the mean | ⚠️ Right symptom, shaky mechanism | (1) Magnitude wrong: row norm is `0.02·√256 ≈ 0.32`; orthogonal init → unit norm, a **~3× increase, not the claimed ~50×** (it computed `1/0.02`). (2) Ignores that `nn.MultiheadAttention` runs queries through an internal xavier projection + zero bias before the dot product, so raw query scale isn't what sets the logits. The right resolution is **empirical** — log attention entropy at init (its own suggestion, and a good one). |
| **Collapse is a training-dynamics problem in the objective; init is polish** | ✅ Correct, and the key point | This is the insight that matters. See `DETAILED_UNDERSTAND.md` §4.2. |
| There's no variance/**covariance** penalty; that's the actual fix | ✅ Mechanism correct | Confirmed: the loss is `flow + 0.10·variance_floor`; no off-diagonal/covariance term. |
| "...the thing your spec called for from day one" | ❌ Wrong for v0.2 | The v0.2 directive **explicitly removed** SIGReg/VICReg/covariance ("...no covariance loss **initially**"). v0.1 had SIGReg. The absence is deliberate, not drift. |
| "It's the attentive pooler, **not a CNN** / there was a CNN→pooler swap" | ❌ Wrong — it's a hybrid | The bottleneck is **both**: real depthwise `Conv2d(7×7)` ConvNeXt blocks **and** the attentive pooler **and** an MLP. No swap; this is the intended v0.2 design. Claude over-extrapolated from a partial screenshot. |

**Net:** keep its conclusion ("collapse lives in the objective; the missing
decorrelation term is the lever") and its zero-init fix. Discard the spec and
CNN-vs-pooler claims. Treat the query-scale fix as cheap hygiene to *verify*,
not assume.

## §2. The decision: VICReg-C first, SIGReg as escalation

The tech lead suggested "maybe let's just try sigreg." We evaluated both
decorrelation options. **Decision: implement the VICReg covariance term
(VICReg-C) as the first experiment; hold SIGReg as the escalation.** Reasoning:

**Why VICReg-C is the better *first* move (not just simpler):**

1. **Targets the exact failure with no collateral.** The diagnosis is "rank
   too low = dims too correlated." VICReg-C penalizes off-diagonal covariance
   — precisely that, one interpretable knob. It can also be computed on the
   same 256-feature covariance the `effective_rank` metric uses, so it acts
   directly on (a smoothed version of) the quantity we want to move.
2. **It does not impose a distributional shape — the clincher.** SSv2 has
   discrete action categories, so the ideal `c_t` may be **clustered /
   multimodal**. SIGReg pushes all 1-D projections toward standard normal,
   which (by Cramér-Wold) pressures `c_t` toward a single **unimodal isotropic
   Gaussian** — it would actively penalize legitimate cluster structure a
   world model may want. VICReg-C only zeroes cross-correlations (a
   second-moment constraint); a mixture of clusters can be globally
   decorrelated, so it **tolerates multimodality**. Forbidding structure we
   don't yet understand is anti-minimalist.
3. **Deterministic gradient.** SIGReg re-draws random projections each step,
   injecting stochastic noise into the regularizer gradient — a minor but real
   consideration right after a gradient-explosion fix.
4. **The theory cited for SIGReg targets a different goal.** LeJEPA's argument
   (isotropic-Gaussian embeddings are provably good for **linear probing**)
   doesn't transfer cleanly: our `c_t` is **conditioning for a generative
   predictor**, not a probe target.

**When SIGReg would win / why we keep it ready:** if we decide the goal is
broader than "fix rank" — a single principled all-in-one collapse guarantee
with isotropy as a desideratum. It's also cheap to resurrect: the v0.1
implementation is in git at **commit `5cb8330`** (`losses.py::sigreg`,
random-projection characteristic-function form). If we use it, apply it to
**`c_t` only** — SIGReg on the frozen `e_t` is pointless (you can't reshape a
frozen distribution with a gradient).

## §3. The reliability / attribution logic (experiment design)

A question we explicitly worked through: *is the regularizer (Fix 2) more
"reliable" than scaling the dataset (Fix 3)?*

- **Yes, as a lever on the rank number.** A regularizer is a direct causal
  force on the exact quantity — crank it and dims decorrelate, rank rises.
  More data is a *hypothesis* whose outcome we can't control.
- **But "moves the metric" ≠ "fixes the real problem."** A regularizer can
  reliably raise rank while making `c_t` no more useful (Goodhart). That's why
  §7 of `DETAILED_UNDERSTAND.md` requires copy-ratio to improve *alongside*
  rank.
- **So the two are not competitors.** Fix 2 = treatment (reliable lever, risk
  of treating a symptom). Fix 3 = diagnosis (tells us whether we needed the
  treatment). **Resolution: fold them together** — run the regularizer on a
  larger/full dataset rather than on tiny, since we need a real run anyway.

**Experiment matrix (the open human/tech-lead decision):**

| Run | Reg | Data | Tells us |
|---|---|---|---|
| Baseline (have it) | none | tiny | rank ~5 reference |
| A | none | full SSv2 | does data alone lift rank? |
| B | VICReg-C | full SSv2 | does the regularizer lift rank (+ copy-ratio) on top of data? |

- **2-run version (A + B):** full attribution (data effect vs reg effect).
- **1-run version (B only):** ship if rank **and** copy-ratio both improve;
  add A only if results are ambiguous. Cheaper, slightly less certain.

**This decision is not yet made** — it's a cost-vs-certainty call for the
human/tech lead. `TASKS.md` is written to support either; the human picks at
the launch handoff (H4.3).

## §4. The init tweaks (Fix 1), settled understanding

Bundle two changes, both flag-gated, understood per §1:

- **Zero-init `out_mlp[-1]`** (weight + bias) → bottleneck starts as
  `abstract = attended` (identity residual). Take as-is.
- **Query init scale** → orthogonal init *or* a larger-scale Gaussian;
  understood as a **scale** correction, verified by the at-init attention-
  entropy diagnostic, not assumed to work.

These "delay degeneracy"; they are hygiene, not the cure. They run first
because they're fast, and the real run (Fix 2 on larger data) proceeds
concurrently with implementing Fix 2.

## §5. What "going forward" looks like (the spine)

1. **Fix 0** — instrument (attention entropy; within-video slot rank vs
   cross-video feature rank). No behavior change. Tells us *which* collapse.
2. **Fix 1** — init tweaks (fast); human runs a short instrumented job while
   the agent implements Fix 2 concurrently.
3. **Fix 2** — VICReg-C behind a flag, run as a larger-data A/B (the main
   event). SIGReg held as escalation.
4. **Fix 4** — frame-overlap / tubelet-horizon task difficulty: **deferred**
   to the Phase 4 conversation.
