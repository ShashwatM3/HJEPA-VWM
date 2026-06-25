# 01 — The Big Picture: World Models, JEPA, and Why We Predict Latents

> **What you'll understand after this file:** why this project exists, why we
> predict compressed representations instead of pixels, what "JEPA" means,
> and why the architecture has *two* levels of latent instead of one.

---

## 1. What is a world model?

A **world model** is a system that learns how the world *evolves*. Given the
current state, it predicts future states. That's it. The term comes from
model-based reinforcement learning (Ha & Schmidhuber's 1992-lineage "World
Models" paper, 2018), where an agent learns a compressed simulator of its
environment and plans inside it.

Why anyone cares: a system that can predict the future has necessarily
learned something real about objects, motion, physics, and causality. You
can't predict where the cup will be after the hand pushes it unless you've
implicitly learned "hands push things" and "things move when pushed."
Prediction is a *training signal for understanding* that requires zero human
labels — the future arrives on its own and tells you whether you were right.

Our project trains a world model on **SSv2 (Something-Something V2)** — a
dataset of ~220k short clips of people doing basic things to objects
("pushing something from left to right", "covering something with
something"). It's the standard benchmark for *temporal* understanding,
because the action matters more than the appearance: you cannot tell
"pushing left-to-right" from "pushing right-to-left" from a single frame.

## 2. The naive approach and why we reject it

The obvious world model: predict the next **frame**. Feed in frames 1–8,
output frame 12, train with pixel-wise MSE.

This fails in two instructive ways:

**Failure 1 — pixels are mostly irrelevant detail.** A 256×256×3 frame is
~200k numbers. The vast majority describe things that don't matter for
dynamics: texture of the tablecloth, exact leaf positions, sensor noise. A
pixel-predicting model spends nearly all its capacity modeling things that
have nothing to do with "what happens next." The important stuff — object
identity, positions, velocities, intentions — is a tiny fraction of the
bits.

**Failure 2 — the future is a distribution, and MSE averages it.** Given 8
frames of a hand reaching toward a cup, several futures are plausible:
grasp, hover, knock over. A deterministic model trained with MSE outputs the
*average* of all plausible futures — a blurry ghost-hand smear. This is the
classic "MSE blur." The model isn't being stupid; the loss literally asks
for the mean. The mean of distinct sharp futures is a blurry non-future.

Both failures point the same direction: **don't predict pixels. Predict a
representation.**

## 3. JEPA: Joint-Embedding Predictive Architecture

JEPA is LeCun's name (2022 position paper, "A Path Towards Autonomous
Machine Intelligence") for the family of architectures that fix this:

1. **Encode** the input into a latent (a learned representation).
2. **Predict** the *latent* of the future/missing part — not its pixels.
3. Train by comparing predicted latent vs actual future latent.

"Joint-embedding" means both the context and the target are mapped into the
same embedding space, and prediction happens entirely inside that space.

The crucial property: **the encoder gets to choose what to ignore.** If
tablecloth texture isn't useful for prediction, the encoder can simply not
represent it, and the prediction loss never punishes that omission. Compare
with pixel prediction, where every ignored detail costs you loss. A JEPA
spends its capacity only on *predictable, future-relevant structure*.

Members of this family you may have heard of: BYOL, SimSiam (images,
augmentation-based), I-JEPA (images, masked-patch prediction), V-JEPA
(video, masked spatiotemporal prediction). Our frozen encoder *is* V-JEPA 2
— we're standing on this exact lineage twice: once for the pretrained
encoder, once for our own architecture's design.

The catch — and it's a big one — is **collapse**: if the encoder produces
the same latent for everything, prediction becomes trivially perfect and
the loss goes to zero while the model learns nothing. The entire
anti-collapse machinery (EMA targets, stop-gradient, variance floor,
diagnostics) exists because of this. Files 05 and 06 are devoted to it.

## 4. Why TWO levels of latent?

Our architecture maintains a hierarchy:

| Latent | Shape | Source | What it captures |
|---|---|---|---|
| `e_t` — detailed | 1024 tokens × 1024 dims | frozen encoder | texture, local appearance, spatial detail |
| `c_t` — abstract | 32 tokens × 256 dims | trainable bottleneck | future-relevant structure: what's happening, where it's going |

Why not just one?

**The argument for compression:** dynamics are low-dimensional. The
*description* of a scene is huge, but the *change* between t and t+k is
governed by a few factors — what's moving, where, how fast. If you predict
in the full detailed space, the predictor must carry along a million bits
of static detail just to say "and the hand moved left." If you predict in a
compact abstract space, the predictor models *only the dynamics*. This is
the "slow feature" / "abstraction" intuition: planning and prediction
happen at a coarse level; detail gets filled in afterwards.

**The bandwidth analogy:** `c_t` is 32×256 = 8,192 numbers. `e_t` is
1024×1024 = ~1M numbers. The bottleneck is a 128× compression. The
hypothesis of this whole project is that ~8k numbers is enough to carry the
*plot* of a short video clip, with `e_t` holding the *cinematography*.

**The hierarchy is testable.** Phase 2 adds a fine predictor `F_e` that
predicts the future `e` *conditioned on* the predicted `c`. The "shuffled-c"
test then checks: if you give `F_e` the *wrong* `c` (shuffled across the
batch), does its prediction get much worse? If yes — the abstract level is
genuinely steering the detailed level, and the hierarchy is real. If no —
`F_e` was ignoring `c`, the hierarchy is decorative, and the design fails
its own contract. This kind of *bypass test* is the project's core
discipline: every architectural claim must have a diagnostic that could
falsify it.

## 5. The full pipeline (intended v0 architecture)

The project builds in phases, but the **target design** is always this stack.
Read it top-to-bottom as information flowing from pixels → abstract dynamics
→ detailed appearance → pixels again.

```
                         ┌──────── FROZEN encoder E (V-JEPA 2 ViT-L/16) ────────┐
context clip x_{≤t}   ──►  E  ──► e_t  (1024×1024 — detailed)                   │
                         └───────────────────────────────────────────────────────┘
                                            │
                                            ▼
                              bottleneck B  ──► c_t  (32×256 — abstract)
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  │                                  │
  F_c: coarse flow                          │  TARGET BRANCH (no grad ever):   │
  z_τ, τ, cond c_t  →  velocity v̂_c        │  future clip ──► E ──► e⁺       │
         │ integrate → ĉ⁺                    │              ──► B_EMA ──► c⁺     │
         ▼                                  │                                  │
  F_e: fine flow [Phase 2]                  │  (e⁺ also target for F_e)        │
  cond (e_t, stopgrad(ĉ⁺)) → ê⁺             │                                  │
         ▼                                  │                                  │
  D: frame generator [Phase 3]              │                                  │
  cond ê⁺ → future frames in VAE space      │                                  │
         └──────────────────────────────────┴──────────────────────────────────┘
```

**Phase 4** adds horizon embedding `h_k` to `F_c` only — same hierarchy,
multiple lookahead distances (file 11).

Each major box maps to a curriculum file:
- frozen encoder → `02-FROZEN-ENCODER.md`
- bottleneck → `03-BOTTLENECK.md`
- all three flow predictors (same math) → `04-FLOW-MATCHING.md`
- EMA target branch → `05-EMA-AND-STOP-GRAD.md`
- variance floor + diagnostics → `06-COLLAPSE-AND-METRICS.md`
- phase schedule + multi-horizon → `11-PHASES-AND-MULTI-HORIZON.md`

### 5b. Phase 1 subset (implemented today)

Phase 1 trains **only the coarse level** — everything above `F_c` in the
diagram. The fine flow, frame generator, and `h_k` do not exist in code yet.

```
                        ┌──────── FROZEN (pretrained V-JEPA 2 ViT-L/16) ────────┐
context clip x_{≤t}   ──►  encoder E  ──► e_t (1024×1024)                       │
                        └───────────────────────────────────────────────────────┘
                                            │
                                            ▼
                              trainable bottleneck B  ──► c_t (32×256)
                                            │
                                            │ (condition)
                                            ▼
        noise z_0, time τ  ──►  flow predictor F_c(z_τ, τ, c_t)  ──► velocity v̂
                                            │
                                            ▼
                       L_flow = ‖v̂ − (c⁺_{t+k} − z_0)‖²     +    λ_var · L_var(c_t)
                                                              (operating: λ_var = 0.5;
                                                               config default: 0.10)

TARGET (no gradients ever flow here):
future clip x_{≤t+k}  ──► same frozen E ──► e⁺ ──► EMA copy B_EMA ──► c⁺_{t+k}
```

Read Phase 1 as: *encode the present, compress it, and learn a generative
predictor that transports noise to the compressed future — using the compressed
present as the steering signal.* Phases 2–3 extend the same flow-matching
recipe to `e⁺` and then to pixels; Phase 4 extends `F_c` to multiple horizons.

## 6. What "success" means for Phase 1

Not "the loss went down." Loss always goes down. Success is defined by
**baselines and gates** (detail in file 06):

1. The predictor must beat the **copy baseline** — predicting "future =
   present". A world model that can't beat "nothing ever changes" has
   learned nothing about dynamics. Gate: model loss ≤ 0.70× copy loss.
2. It must beat the **batch-mean baseline** — predicting the average future.
   Beating this proves predictions are *input-specific*. Gate: ≤ 0.50×.
3. `c_t` must be **healthy**: variance alive, videos distinguishable,
   effective rank high enough that the latent actually uses its capacity.

These gates are Phase 1's definition of "the coarse level works." Later
phases add their own gates (shuffled-c for `F_e`, pixel quality for `D` —
file 11). A beautiful loss curve that fails the gates is a failure.

## 7. Mental model to carry forward

Keep this picture: the project is a **bet** that

> *a 128×-compressed latent, trained only to be predictable-from-the-past
> and predictive-of-the-future, will spontaneously organize into a useful
> abstract state — provided we stop it from cheating (collapse) and verify
> it isn't decorative (bypass tests).*

Every design choice in the next files — freezing the encoder, the EMA
target, the variance floor, the diagnostics — is either (a) making the bet
cheaper to test, or (b) closing a loophole through which the model could
"win" without learning. When you evaluate any proposed change, ask: which
of these two jobs is it doing?
