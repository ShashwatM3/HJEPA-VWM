# 10 — Failure Modes: Run 1 as a Case Study in How Training Dies

> **What you'll understand after this file:** the complete mechanical chain
> of a gradient explosion — from "one large batch" to "NaN everywhere" to
> "crash in a diagnostic" — plus the general taxonomy of training failures
> and the defense-in-depth that now guards against them. This file is the
> distilled, teach-it-properly version of
> `AGENT_FILES/KANBAN/02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md`.

---

## 1. The timeline (run `peachy-terrain-5`, 2026-06)

| Step | Observation |
|---|---|
| 0 – ~10,000 | Healthy. loss ~1.09, grad_norm ~3–8 (pre-clip), L_var ~0.085, all latent metrics stable |
| ~10,550 | LR multiplier near its **peak** (end of the 10k warmup of the original schedule) |
| 10,700 | `grad_norm = 3.5 × 10¹⁶` — sixteen orders of magnitude above normal |
| 10,750 | `loss = nan`, `L_flow = nan`, `L_var = nan`, `grad_norm = nan` |
| ~11,000 | Diagnostic tick: `effective_rank` calls `torch.linalg.eigvalsh` on a NaN covariance → `torch._C._LinAlgError` → **process dies** |

Note the deaths: the *model* died at ~10,700; the *process* died at
~11,000, in a diagnostic, with an error message about eigenvalues that had
nothing to do with the root cause. Production failures routinely present
this way — **the crash site is downstream of the crime scene.** The
discipline is to walk the logs *backwards* from the traceback to the first
anomalous number. Here that was grad_norm — which is exactly why it's
logged every 50 steps.

## 2. The mechanism, link by link

**Link 1 — a large gradient at the worst moment.** Some batch produced an
unusually large gradient. Random τ draws, an odd clip, an unlucky noise
sample — individually routine. What made it lethal was *timing*: the LR
schedule was at peak (2e-4/4e-4, the original values), so the parameter
step taken on that gradient was the largest the run would ever take.

**Link 2 — clipping helped less than it appears.** With `grad_clip = 1.0`,
the update was rescaled to norm 1.0 — but two leaks remain. First, Adam's
second moment `v` ingests the (clipped) gradient and with β₂ = 0.95 its
memory is only ~20 steps, so the calibration that *divides* upcoming
updates adapts quickly — and a brief spike both distorts it now and is
forgotten fast, removing the dampening buffer. Second and more
fundamental: a clip preserves the *direction*, and a direction computed
from a pathological batch can point somewhere bad. Repeated at peak LR,
the weights walked into a sharp region of the loss surface.

**Link 3 — bf16 turned "large" into "garbage."** In a sharp region,
activations and gradients span huge dynamic ranges. bf16 has fp32's
*range* (nothing overflows to Inf at first) but only ~3 significant
digits. Intermediate computations started rounding catastrophically —
think `(a + tiny) − a = 0` where tiny was the actual signal. Gradient
*directions* became noise before any value became non-finite. The
explosion fed on its own imprecision: garbage step → sharper region →
bigger gradients → more garbage. By 10,700 the norm was 10¹⁶.

**Link 4 — NaN, the absorbing state.** Eventually an operation produced
`Inf` (overflow of even bf16's range, or a division by ~0), and then
`Inf − Inf = NaN`. NaN's algebra makes it absorbing: anything touching NaN
is NaN. One NaN parameter → NaN activations everywhere → NaN loss → NaN
gradients → *every* parameter NaN after one more update. A model does not
"partially" go NaN for long. There is no recovery except restoring from a
checkpoint — by 10,750 the run was over, whether or not the process knew.

**Link 5 — the diagnostic delivered the coup de grâce.** At the next
500-step diagnostic tick, `effective_rank` built a covariance matrix of
NaNs and handed it to an eigendecomposition, which threw. The training
loop had no handler. The irony worth remembering: the *monitoring system*
crashed the patient. A diagnostic must never be able to take down the
thing it monitors.

## 3. What was missed beforehand (the honest accounting)

1. **Peak-LR risk wasn't treated as a milestone.** The original milestone
   schedule checked "shortly after warmup" generically, but peak LR is
   *the* maximum-energy moment of a run and deserved a dedicated check
   with a go/no-go criterion on grad_norm trend.
2. **grad_norm trend was visible but unalarmed.** Pre-clip norms of 3–8
   against a clip of 1.0 means *every step was being clipped* — the model
   "wanted" much larger steps all along. That standing tension was a
   warning sign in plain sight.
3. **The 30k/10k schedule was calibrated for the wrong dataset** (full
   SSv2), making everything about the run — including where peak LR landed
   relative to epochs of repeated data — different from the spec's
   assumptions (file 07 §5).
4. **No last line of defense.** Clipping was the only guard; there was no
   "this gradient is absurd, learn nothing from it" circuit breaker.

## 4. The fixes, mapped to the links they cut

| Fix | Cuts link | Reasoning |
|---|---|---|
| Peak LRs halved (1e-4 / 2e-4) | 1 | less energy at the most dangerous moment; "survives warmup, dies at peak → halve the peak" |
| `grad_clip` 1.0 → 0.5 | 2 | smaller worst-case step; also keeps bf16 rescaling division better-conditioned |
| `grad_skip_threshold = 50` + skip-guard (no step, no EMA, log `grad_skipped`) | 2→3 | absurd gradients carry no information; discard the step entirely rather than rescale garbage |
| EMA skipped on skipped steps | 4 | a corrupted online module must not leak into the target via averaging |
| `effective_rank` returns NaN on non-finite covariance | 5 | a NaN model is a *logged finding*; the loop survives to checkpoint and report |
| 15k steps, warmup 1.5k, checkpoints every 2.5k | context | right-sized to ssv2_tiny; bounded loss-on-crash (~50 min) |

This is **defense in depth**: no single fix is trusted to be sufficient;
each link in the causal chain gets its own cut. Also note what was *not*
done: no switch to fp32 (cost), no removal of bf16 (the guards make it
safe enough), no blind LR sweep (the postmortem pinpointed the mechanism,
so the fix could be targeted).

## 5. The general taxonomy (beyond Run 1)

A reference card of the ways training dies, so you can pattern-match:

| Failure | Signature | First response |
|---|---|---|
| **Gradient explosion** | grad_norm jumps orders of magnitude; loss spikes then NaN | lower peak LR, tighter clip, skip-guard; check for data outliers |
| **NaN from bad math** (no explosion) | sudden NaN with calm grad_norm | hunt the op: log/sqrt/÷ of ~0, masked softmax over nothing; clamp/eps at the site |
| **Loss plateau** | flat loss from early on | LR too low, bug in gradient path (assert grads exist!), target trivially satisfied |
| **Collapse family** | great loss, dead diagnostics | file 06 — this is why the panel exists |
| **Overfitting/overtraining** | train loss falls, val diagnostics worsen; too many epochs of a small set | more data, fewer steps, stronger augmentation — *re-derive epochs whenever dataset changes* |
| **Silent data corruption** | metrics fine, results subtly wrong | contract tests, range asserts (e.g. the not-[-1,1] normalization check) |
| **Infra death** (OOM, preemption, disconnect) | process gone, no model pathology | tmux, checkpoints, resume — file 09 |

The cross-cutting habits that make all of these survivable: **log cheap
scalars frequently** (the autopsy is only as good as the flight recorder);
**checkpoint at the granularity of acceptable loss**; **make monitors
crash-proof**; and **write the postmortem** — the document, not the
feeling of having understood. Run 1 cost ~$10 and a day; the postmortem
converted it into permanent schedule-sizing rules, two new guards, and
this file.

## 6. Questions to test yourself

1. Why did the run die at ~10,550 of all places? *(End of the original
   10k warmup — peak LR; the largest steps of the run meet a model good
   enough to have sharp loss-surface regions.)*
2. Explain how `Inf − Inf` arises in a flow-matching loss. *(Overflowed
   activations propagate to û and u; their difference is Inf − Inf =
   NaN.)*
3. Why does the skip-guard zero the gradient instead of clipping harder?
   *(Direction from a pathological batch is information-free; any step
   along it — however small — is noise injection, and the bf16 rescale
   adds rounding damage.)*
4. Why was every-step clipping at norms 3–8 a warning sign? *(The model
   persistently wanted 3–8× larger steps than allowed — standing tension
   between LR and loss-surface sharpness, waiting for a trigger.)*
5. What's wrong with a diagnostic that can throw? *(It couples the
   monitor's failure to the patient's: you lose checkpointing, logging,
   and a clean final state precisely when you most need them.)*

---

## 7. Case study 2: the grad-skip death spiral (`elated-snowflake-15`)

Run 1 taught us about gradient *explosions*. Investigation 005's
`elated-snowflake-15` taught a different lesson: the skip-guard can
**freeze training while metrics still look fine.**

### Timeline

| Step | Observation |
|---|---|
| 0 – ~8000 | Healthy. Matches `cerulean-snow-13`: copy ratio 0.83–0.96, rank ~13.7, std ~1.04, `grad_skipped=0` |
| ~8500 | Single spike: `grad_norm=170`, `grad_skipped=1`, copy ratio jumps to 5.7 |
| 8500 – ~13850 | **Every step skipped.** Weights frozen. Copy ratio stuck 3.5–5.7. |
| ~13850 | Process ends. ~5300 steps of zero learning after the spike. |

### The mechanism

1. One batch at step ~8500 produces a pre-clip gradient norm of 170 —
   above the skip threshold (50) but not a full NaN explosion.
2. The skip-guard correctly discards the step. But unlike Run 1, the
   weights do **not** go NaN — they stay in a bad region of the loss
   surface.
3. Every subsequent batch also produces huge gradients (norm 100–250).
   The skip-guard fires on **every** step. The optimizer never updates.
   EMA never updates (by design — file 07).
4. **The latent metrics lie.** Forward passes still run; `c_std_mean`,
   `c_cross_video_cosine`, and `c_effective_rank` look healthy (~0.17,
   ~13.7, ~1.10) because they measure the *current* weights, not whether
   those weights are *improving*. Copy ratio — which re-scores prediction
   quality — reveals the truth: 3.5–5.7 means the model is worse than
   "do nothing."

### What this teaches

| Lesson | Detail |
|---|---|
| **`grad_skipped` is a hard gate** | Any sustained nonzero rate means stop immediately — not "the guard is handling it" |
| **Latent health ≠ training health** | Diagnostics on frozen weights can look fine while optimization is dead |
| **Copy ratio is the canary** | When in doubt, trust the baseline ratios over variance/rank |
| **Resume strategy matters** | Do not resume from the final checkpoint (garbage). Resume from **pre-spike** (~step 7500) with lower flow LR (`--lr-coarse-flow 1e-4`) |
| **Different failure, same defense** | Run 1 needed lower peak LR; this run needed earlier intervention when skip rate went nonzero |

This is why the health checklist in file 09 lists `grad_skipped == 0`
as the first line — before any latent metric.
