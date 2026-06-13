# 04 — Detailed understanding: Fixing dimensional collapse of `c_t`

> **Read order for this folder:**
> 1. This file — what the problem is, the evidence, and why it happens.
> 2. [`ANALYSIS_AND_DECISIONS.md`](ANALYSIS_AND_DECISIONS.md) — the graded
>    review of the external opinions and the locked design decision
>    (VICReg-C first, SIGReg as escalation) with full reasoning.
> 3. [`EXECUTION_PHASES.md`](EXECUTION_PHASES.md) — the run-gated roadmap
>    (P1→P4, with a pipeline run between phases). **The map.**
> 4. [`TASKS.md`](TASKS.md) — the agent's sequenced, falsifiable `A4.x` plan.
> 5. [`HUMAN_TASKS.md`](HUMAN_TASKS.md) — the pod/W&B work only the human can do.
>
> **Background (read if you lack context):**
> [`../../../YOUR_FILES/LEARN/06-COLLAPSE-AND-METRICS.md`](../../../YOUR_FILES/LEARN/06-COLLAPSE-AND-METRICS.md)
> (collapse taxonomy + the five metrics) and
> [`../../../YOUR_FILES/LEARN/03-BOTTLENECK.md`](../../../YOUR_FILES/LEARN/03-BOTTLENECK.md)
> (bottleneck internals, §7 on this exact rank problem).

---

## §1. One-paragraph statement of the problem

In Phase 1 Run 1 (`peachy-terrain-5`), the abstract latent `c_t` showed
**dimensional collapse**: `c_effective_rank` plateaued at **~5 out of a
maximum 256** and stayed there for the entire ~10k healthy steps before the
unrelated gradient-explosion crash. The bottleneck is using ~2% of its
representational capacity. The other collapse metrics were fine (variance
alive, cross-video cosine healthy), so this is **not** full collapse or
directional collapse — it is specifically the low-rank-subspace failure.
This task fixes it.

## §2. Two separate failures — do not conflate them

| Failure | Type | Status | Where it's documented |
|---|---|---|---|
| Crash at step ~10,750 | Gradient explosion (optimization) | **Fixed** (lower LR, tighter clip, skip-guard) | [`../02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md`](../02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md) and [`LEARN/10-FAILURE-MODES.md`](../../../YOUR_FILES/LEARN/10-FAILURE-MODES.md) |
| Rank ~5 plateau | Dimensional collapse (representation) | **Open — this task** | this folder |

The crash was loud and is solved. The collapse is quiet and is the real
research problem. Everything in this folder is about the second row.

## §3. The evidence

From Run 1 W&B (`smahalanobis-uc-davis/hjepa-vwm/runs/1chv2608`), during the
healthy phase (steps 0–~10,000):

- `c_effective_rank` ≈ **5**, flat (healthy spec is **> 60**; concerning < 20;
  hard-stop < 5). It was not a transient — it never climbed.
- `c_std_mean` ≈ 1.0, `c_dead_dim_frac` ≈ 0 → **not** full collapse.
- `c_cross_video_cosine` well below 0.5 → **not** directional collapse.
- Acceptance ratios were improving slowly but never reached the gates before
  the crash.

Conclusion: per-dimension variance is healthy, videos are distinguishable,
but the representation lives in a ~5-dimensional subspace. The 256 feature
dimensions are heavily **correlated** — effectively ~5 underlying factors
copied across many dims.

## §4. Why this happens — the mechanism (important)

### §4.1 The variance floor is structurally incapable of fixing it

The only explicit anti-collapse term is the variance floor:

```85:107:losses.py
def variance_floor(abstract: Tensor, std_target: float = 1.0) -> Tensor:
    ...
    std = flat.std(dim=0, unbiased=False)
    return torch.clamp(std_target - std, min=0.0).mean()
```

It hinges the **per-dimension standard deviation** at 1.0. That constrains
the **diagonal** of the covariance matrix only. It says nothing about the
**off-diagonal** entries. So a latent can satisfy the floor perfectly (every
dim has unit std) while every dimension is a near-copy of ~5 underlying
directions — unit variance on the diagonal, near-1 correlations off it,
effective rank ~5. **The variance floor cannot see or prevent this.** This
is not a tuning issue; no value of `lambda_var` fixes it, because the term
has no gradient w.r.t. off-diagonal correlation.

### §4.2 Collapse is decided by the objective, not the initialization

The flow-matching loss is happiest when its target `c⁺` is *easy to
predict*, and a low-rank target is easier. The EMA self-distillation loop
therefore applies a steady pressure toward shedding dimensions, which the
variance floor does not counteract. Initialization sets the *starting* rank;
training dynamics decide the *ending* rank. The ~5 plateau over 10k steps is
evidence the optimizer found and held a low-rank optimum — an objective
problem, not an init transient.

### §4.3 A metric caveat the fix must account for

`effective_rank` pools all 32 query-slots together as samples:

```91:100:diagnostics.py
    flat = abstract.float().reshape(-1, abstract.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.t() @ flat / max(1, flat.shape[0] - 1)
    ...
    return {"c_effective_rank": float(torch.exp(entropy).item())}
```

So a low value conflates **slot redundancy** (all 32 query outputs nearly
identical) with **feature-dim correlation** (the 256 dims correlated). These
have different fixes. The instrumentation task (A4.0) must split them before
we commit to a remedy.

## §5. Candidate causes (each has a different fix; we test, not guess)

1. **Missing decorrelation regularizer (objective).** §4.1–4.2. The primary
   hypothesis. Fix: add a covariance/decorrelation term (VICReg-C; SIGReg as
   escalation). See `ANALYSIS_AND_DECISIONS.md`.
2. **Data-limited.** Rank ~5 was measured on `ssv2_tiny` (~4k clips, ~240
   epochs at 15k steps). The subset may genuinely have ~5 axes of variation.
   Fix/test: run on full SSv2 and see if rank rises on its own.
3. **Initialization artifact.** Near-uniform cross-attention at init (tiny
   query scale) + non-identity residual MLP. Fix: query init scale +
   zero-init `out_mlp` last layer. Understood as *delaying* degeneracy, not
   curing it.
4. **Task too easy (frame-overlap).** §6. Contributes weak pressure on `c_t`.
   Deferred (Fix 4 / Phase 4 territory).

## §6. Tubelet verification (a recurring tech-lead question — answered)

**Are we operating in tubelet space?** At the representation level, **yes**.
The encoder geometry (`config.py` `n_ctx`) is `(T/2)·(H/16)² = 4·256 = 1024`
tokens: tubelet-2 merges 8 frames into **4 temporal tubelets** × 256 spatial.
Every `e_t`/`c_t` token is a 2-frame spatiotemporal tubelet — not a single
frame. The tech lead's "encode spatiotemporal, not spatial" requirement is
already satisfied by the frozen encoder.

**The real subtlety:** the prediction *horizon* is frame-defined and the
context/target windows **overlap ~75%**. With `t_ctx=8, stride=2, k=4`:
context covers frames `start … start+14`, target covers `start+4 … start+18`
— the target is the context slid forward by **2 of 4 tubelets**. Predicting
a near-identical future needs few dimensions of "what changed," which is a
plausible *contributor* to low rank. The tech lead's "predict k tubelets
ahead" idea (less overlap, tubelet-aligned offset) is the fix for this, and
it lands naturally in Phase 4 multi-horizon — **deferred, not now.**

## §7. Success criteria for this task

A fix is accepted only if **both** hold (guarding against Goodhart — raising
the rank number while degrading the model is not success):

1. `c_effective_rank` rises materially — target **> 30** on the 15k scale
   (toward the > 60 spec), with `c_dead_dim_frac` staying < 0.15.
2. Prediction quality does **not** regress — `coarse_vs_copy_ratio` improves
   (or at minimum holds) alongside the rank increase.

Plus the standard guards stay green: no NaN, `grad_skipped` = 0, frozen
encoder unchanged, `c_cross_video_cosine` well below 0.5.

## §8. Scope and guardrails

- **Phase 1 v0.2 only.** No encoder change, no Phase 2+ work.
- **This exercises the supervisor's "initially" clause.** The v0.2 directive
  was *"only a variance floor on c_t, no SIGReg, no full VICReg, no
  covariance loss initially"*
  ([`SUPERVISOR_FEEDBACK_EXPLAINED.md`](../../KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md)
  §7.2). Adding a covariance term is a deliberate, supervisor-aware escalation
  prompted by the tech lead ("maybe let's just try sigreg"). Keep the change
  **flag-gated and reversible** so the variance-floor-only baseline is always
  one config switch away.
- **Every change behind a config flag, default off**, so runs are clean A/Bs
  and attribution is possible.
