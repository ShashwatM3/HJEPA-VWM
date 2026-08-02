# 05 — EMA Targets and Stop-Gradient: Why the Target Comes From a Slow Twin

> **What you'll understand after this file:** the self-supervised
> chicken-and-egg problem, what an EMA (exponential moving average) of
> weights is, why the target bottleneck is an EMA copy rather than the live
> module or a fixed one, what the momentum schedule does, and where every
> gradient boundary lives in the code.

---

## 1. The chicken-and-egg problem of self-supervised targets

Supervised learning has fixed targets (labels). We don't. Our target —
"the abstract latent of the future clip," `c⁺` — is *produced by the very
model we are training*. That circularity creates two opposite failure
modes, and the EMA design is the engineered compromise between them.

**Option A: use the live bottleneck for the target too.** Then the loss is
`‖F_c(...) − stuff(B(e⁺))‖²` where `B` appears on both sides. Even with
gradients blocked on the target side, the *target moves every step exactly
as fast as the model*. Two problems:

1. **Moving-target instability:** the predictor chases a target that
   changes with every optimizer step — like learning to hit a dartboard
   someone re-hangs after each throw. Optimization becomes noisy and can
   oscillate.
2. **Cooperative collapse:** the bottleneck controls both the conditioning
   *and* the target. The system-wide easiest way to make the loss small is
   for `B` to make targets *predictable* — and nothing is more predictable
   than a constant. The live-target configuration gives the model the
   shortest path to degenerate solutions.

**Option B: fix the target bottleneck forever (random or pretrained-ish).**
Stable, but now `c⁺` is defined by a *random projection* of `e⁺` that never
improves. The online bottleneck learns to predict a frozen, meaningless
embedding. The target can never get smarter along with the model.

**Option C (ours): the target is a slow-moving average of the online
weights.** The target improves as the model improves — but smoothed over
thousands of steps, so it's stable on the timescale of individual updates.
You get Option A's adaptivity with most of Option B's stability. This is
the BYOL / I-JEPA / V-JEPA recipe, and it is empirically what makes
negative-free self-supervised learning work at all. (Fun fact from the
literature: BYOL's authors found that *without* the EMA/stop-grad
asymmetry, representations collapse; SimSiam showed stop-grad alone can
suffice in some setups. The asymmetry between the two branches is the
load-bearing element.)

## 2. What EMA-of-weights actually is

Every parameter of the target bottleneck is updated, once per training
step, as:

```
θ_target ← m · θ_target + (1 − m) · θ_online        m ≈ 0.996 … 0.9999
```

In code (`train.py::_update_ema`), via `lerp_(p_online, 1.0 - momentum)` —
the same formula rearranged. No gradients, no optimizer; pure averaging.

Intuition for the number: with `m = 0.996`, each update keeps 99.6% of the
old value. The target is effectively an average of the online weights over
the last `1/(1−m) = 250` steps. With `m = 0.9999`, that horizon is 10,000
steps. So the momentum directly sets *how blurred in time* the target is.

### The momentum schedule

```python
ema_m_start = 0.996      # target follows within ~250 steps
ema_m_end   = 0.9999     # target follows within ~10,000 steps
# cosine ramp between them over ema_schedule_steps = 105,000
```

Why ramp it (BYOL's recipe, kept by everyone since):

- **Early training:** online weights are nearly random and improving fast.
  A sluggish target would pin the model to a long average of *garbage*.
  Low momentum lets the target track quickly while everything is cheap to
  change.
- **Late training:** representations are good; what you want is target
  *stability* so the predictor can converge against a near-fixed
  objective. High momentum makes the target nearly frozen.

A subtlety worth knowing: the schedule denominator is 105,000 (the full
multi-phase latent-training horizon), while Phase 1 runs only 15k steps —
so within Phase 1 we only traverse the early part of the cosine ramp
(m ≈ 0.996 → ~0.9962). That's intentional: the schedule belongs to the
whole curriculum, not to one phase.

Also note the interaction with the skip-guard (file 07): on a skipped step
(exploded gradient), the EMA update is *also* skipped — otherwise a
corrupted online module would still leak into the target through
averaging.

## 3. Stop-gradient: the other half of the seal

EMA controls how the target's *weights* evolve. Stop-gradient controls
where *gradients* may flow at run time. Both are needed; they answer
different questions.

The rule in this codebase: **the target branch is triple-sealed.**

1. `requires_grad = False` on every `TargetBottleneck` parameter
   (`freeze()`), so autograd never computes grads for them.
2. The forward runs under `torch.no_grad()`, so no graph is even built.
3. The output passes through `as_target()` — a named wrapper around
   `.detach()`:

```32:44:losses.py
def as_target(x: Tensor) -> Tensor:
    """Mark a tensor as a stop-gradient target.

    Used for the EMA bottleneck (`B_EMA`) output, the frozen-encoder target latent,
    and detached conditioning, so every gradient boundary is visible through one
    named helper instead of scattered `.detach()` calls.
    """
    return x.detach()
```

The naming is an MLOps discipline point: gradient boundaries are the most
dangerous invisible structure in an ML codebase. Wrapping `.detach()` in a
function called `as_target` makes every boundary *searchable and
reviewable*. When someone asks "could gradients leak into the target?",
the answer is `grep as_target` plus one smoke-test assertion:

```410:412:models.py
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert any(p.grad is not None for p in coarse_flow.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
```

**Why must gradients never reach the target?** Because the gradient of
`‖û − u‖²` with respect to the *target* points toward "move the target
closer to the prediction." Let that flow and the model improves the loss
by making targets easier instead of predictions better — the formal
definition of cheating. Collapse is this cheat taken to its limit.

## 4. The complete gradient map of Phase 1

Worth having in your head as a picture (**Phase 1 gradient routing** — Phase 2
adds `F_e`, Phase 3 adds `D`; encoder and target-branch rules stay the same):

```
                                 GRADIENTS FLOW          GRADIENTS BLOCKED
context_clip ─► E (frozen) ─► e_t          ─┐            E: no_grad, frozen
                                            ▼
                              B (online) ─► c_t ─► F_c conditioning ─► L_flow
                                            │                              ▲
                                            └────► L_var                   │
                                                                           │
target_clip ─► E (frozen) ─► e⁺ ─► B_EMA ─► c⁺ ─(detach)─► z_τ, u ─────────┘
```

Trainable-by-gradient: **B and F_c only.** Updated-without-gradients:
**B_EMA** (by EMA). Never updated: **E**. Three different "frozen-ness"
regimes in one model — keep them distinct in your vocabulary:

| Module | Params change? | How | Receives grads? |
|---|---|---|---|
| E (encoder) | never | — | never |
| B (bottleneck) | every step | AdamW | yes |
| B_EMA (target) | every step | EMA from B | never |
| F_c (flow) | every step | AdamW | yes |

(Answering a confusion you hit once: "the bottleneck is frozen" is *false*
for B and *true-but-misleading* for B_EMA — B_EMA's weights move every
step, just not by backprop.)

## 5. v0.2 simplification: why there is no target *encoder*

In standard V-JEPA/BYOL, the EMA copy is of the *encoder* — the biggest
collapsible module. In our v0.2, the encoder is frozen and shared by both
branches; the only trainable module on the target path is the bottleneck.
Hence: **EMA on the bottleneck only.** One EMA module instead of two, no
"target encoder" concept at all, and the only collapse-capable component
(the trainable bottleneck) is exactly the one wrapped in the EMA+stop-grad
machinery. The design rule generalizes: *put the EMA where the trainable,
collapse-capable parameters are — nowhere else.*

Lifecycle details that matter for correctness:

- **Initialization:** at startup, `copy_weights_from` makes B_EMA an exact
  clone of B, so step 0 targets come from the same function as the online
  branch (just without grads).
- **Checkpointing:** B_EMA's state dict is saved and restored. If you
  resumed without it, the target would snap back to whatever B is at
  resume — losing the smoothing history. (This is a classic resume bug in
  EMA codebases; ours saves it — see `save_checkpoint`.)
- **Eval pinning:** like the encoder, B_EMA overrides `train()` to stay in
  eval mode forever.

## 6. Why this matters even with a frozen encoder (the honest question)

You might object: "If E is frozen, e⁺ is stable; why does the target
bottleneck need EMA at all — couldn't we use the live B with just a
detach?" This is a legitimate research question. Arguments for keeping
EMA:

- Without it, the target still moves at full speed (problem 1 of Option
  A), making the predictor's objective noisier.
- The bottleneck *is* trainable and could still drift toward
  easy-to-predict (low-information) outputs faster than the variance floor
  can push back; EMA slows that feedback loop.
- It's cheap (one extra 1.7M-param module, a `lerp_` per step) and it's
  the battle-tested recipe.

But note the supervisor's design philosophy cuts both ways — v0.2 removed
SIGReg for minimalism while *keeping* EMA. The implicit judgment: EMA is
load-bearing, covariance regularization is not (yet). If a future ablation
showed detach-only works equally well here, that would be a real
simplification. Until tested, we keep the proven recipe.

## 7. Questions to test yourself

1. Write the EMA update from memory. What horizon does m=0.996 imply?
   *(θ_t ← 0.996·θ_t + 0.004·θ_o; ~250-step average.)*
2. Why is the EMA update skipped when a gradient step is skipped? *(A
   corrupted/exploded online module must not leak into the target via
   averaging.)*
3. Three mechanisms seal the target branch — name them. *(requires_grad
   False, no_grad forward, as_target detach.)*
4. Why is there no target encoder in v0.2? *(Encoder is frozen and shared;
   EMA exists to stabilize *trainable* target-path modules, and only the
   bottleneck qualifies.)*
5. If you let gradients flow into the target, what does the model learn to
   do? *(Move targets toward predictions — degrade target informativeness
   instead of improving predictions; collapse in the limit.)*
