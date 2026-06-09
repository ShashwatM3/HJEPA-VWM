# SUPERVISOR_FEEDBACK_EXPLAINED.md — The flow-matching / frozen-encoder / multi-horizon update, explained from first principles

> **What this file is.** A standalone teaching document. It explains, in full and from the ground
> up, the architecture update the supervisor sent. It assumes you have *just* learned the JEPA and
> flow-matching vocabulary and does not assume you can fill in gaps. Every term is defined where it
> first appears.
>
> **Why it exists separately.** The change is conceptual, not just numerical. If you only patch the
> constants table you will get the *what* without the *why*, and the why is what stops you from
> reintroducing the exact failure modes this design is built to avoid. Read this once, fully, before
> reading [`BRIEF_V0_2.md`](BRIEF_V0_2.md) or the updated [`UNDERSTANDING.md`](UNDERSTANDING.md).
>
> **Precedence.** Where this document (sourced from the supervisor's message) conflicts with
> [`BRIEF_V0_1.md`](BRIEF_V0_1.md) (the original PDF), **the supervisor's update wins.** See
> [`ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md`](ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md) for the formal
> precedence order.

---

## 0. The supervisor's message, verbatim

> We are updating Phase 1 to use a conditional rectified-flow / flow-matching predictor over the
> abstract latent (c_t), not a deterministic predictor. Use a frozen pretrained ViT encoder (E) to
> produce (e_t = E(x_{≤t})), then a trainable bottleneck/CNN (B) to produce (c_t = B(e_t)). For a
> target horizon (k), compute the EMA/stop-grad target (c⁺_{t+k} = B_EMA(E(x_{≤t+k}))). Sample
> (z_0 ∼ N(0,I)), (τ ∼ U(0,1)), form (z_τ = (1−τ)z_0 + τ c⁺_{t+k}), and train the predictor
> (v_θ(z_τ, τ, c_t, h_k)) to match the velocity target (v_target = c⁺_{t+k} − z_0) using MSE:
> (L_flow = ||v_θ − (c⁺_{t+k} − z_0)||²).
>
> We are also switching to multi-horizon prediction. For each sample, choose (k ∈ {4,8,16,32})
> frames, preferably sampled with probabilities ({0.30, 0.30, 0.25, 0.15}), and pass a learned
> horizon embedding (h_k) into the same shared predictor. Collapse prevention should be minimal: use
> only a variance floor on (c_t), no SIGReg, no full VICReg, no covariance loss initially. Flatten
> (c_t) across slots/features per batch, compute per-dimension std, and add
> (L_var = (1/d) Σ_j max(0, 1.0 − Std(c_j))). Total loss: (L = L_flow + 0.1 L_var). The variance
> floor is only to prevent constant (c_t); the temporal flow objective should learn the actual
> semantics/dynamics.
>
> And then things we need to log/monitor AT MINIMUM: variance of c_t; cosine similarity of c_t from
> one vid to another; effective rank of c_t.
>
> To be very clear, new pipeline is: data → pretrained ViT → e_t → CNN bottleneck → c_t. Flow match
> to get c_(t+k). Do variance regularization by setting a variance floor on c_t to prevent collapse.
> Keep EMA target when encoding the target for flow matching. Whatever pretrained ViT we use should
> be from some sort of world model architecture like DINO or V-JEPA or similar.

The rest of this document unpacks every clause.

---

## 1. The one-paragraph summary, in plain words

We are no longer training our own video encoder from scratch. We take a big **pretrained** vision
transformer that already understands video (from the DINO / V-JEPA family), **freeze** it, and use
its output as our detailed latent `e_t`. The only thing we train on top is a small **bottleneck**
that squeezes `e_t` into the abstract latent `c_t`. Then — exactly as before — we learn to
**predict the future abstract latent** with a flow-matching network. Two things become richer than
before: (1) we predict the future at **several time horizons** (4, 8, 16, 32 frames ahead), not just
one step, using one shared predictor told which horizon it is aiming at; and (2) we throw away the
heavier SIGReg collapse-prevention machinery and replace it with the **simplest possible**
anti-collapse term — a floor on the per-dimension standard deviation of `c_t`.

That is the whole change. The rest is detail.

---

## 2. "Conditional rectified-flow / flow-matching predictor, not a deterministic predictor"

### 2.1 What a *deterministic* predictor would be

A deterministic predictor is a network that takes the current state and outputs **one** answer for
the future state directly: `c_hat = g(c_t)`, trained with `||c_hat − c_plus||²`. One input, one
output, no randomness. The word "deterministic" means: run it twice on the same input, get the same
output.

### 2.2 Why that is a problem for predicting the future

The future is **not** a single point. Given 4 frames of a hand approaching a cup, several futures
are plausible (the hand grasps, hovers, knocks it). If you force a deterministic network to output
one vector with an MSE loss, it learns the **average** of all plausible futures. Averages of
distinct possibilities are blurry, meaningless midpoints — the classic "MSE blur." For a world
model whose entire job is to represent *what could happen next*, regressing to the mean is a
silent failure.

### 2.3 What a *flow-matching* predictor is instead

A flow-matching predictor learns to **generate** the future state from noise, rather than output it
directly. Think of it as learning a current of water (a "velocity field") that, if you drop a
random speck of noise into it and let it drift for one unit of time, carries that speck to a
plausible future state. Because you start from a *different* random speck each time, you can sample
*different* plausible futures. It models the whole distribution, not the mean.

"Rectified flow" is the simplest, straightest version of this idea: the path from noise to target
is a **straight line**, and the velocity along that line is **constant**. That is the only
flow-matching variant we use. (Reference: Lipman et al., *Flow Matching for Generative Modeling*,
arXiv:2210.02747; Liu et al., rectified flow.)

### 2.4 "Conditional"

"Conditional" means the velocity field is **steered by the present**. The predictor doesn't just
generate *any* plausible future abstract state; it generates the future state that is plausible
**given `c_t`** (and, now, given the horizon `h_k`). `c_t` is the condition. This is what makes it a
*predictor* and not just a generator.

### 2.5 Important: we already do this

Our current architecture's coarse flow `F_c` (see [`BRIEF_V0_1.md`](BRIEF_V0_1.md) §4.1) is
**already** a conditional rectified-flow predictor over `c_t`. So this clause is mostly the
supervisor **confirming** the design and re-stating it in cleaner notation. The genuinely new parts
are the **frozen encoder** (§3), the **multi-horizon** generalization (§5), and the **variance
floor** (§6). What changes about the flow itself is only that the *same* predictor is now also
conditioned on a horizon embedding `h_k`.

---

## 3. "Frozen pretrained ViT encoder (E) → e_t, then a trainable bottleneck/CNN (B) → c_t"

This is the biggest change.

### 3.1 What we had before

In `BRIEF_V0_1.md`, the encoder `E` was a small VideoViT-Small (12 layers, dim 384) **trained from
scratch** on our data. The target encoder `E_bar` was an EMA copy of it, also from scratch. We were
asking ~22M parameters to learn good video features *and* the predictive structure *and* avoid
collapse, all at once, on one GPU and a modest dataset.

### 3.2 What we do now

We replace the from-scratch encoder with a **pretrained** ViT that already learned strong video
features on internet-scale data, and we **freeze** it — `requires_grad = False`, never updated. We
keep only a small **trainable** module on top: the bottleneck `B` (the "CNN" the supervisor refers
to — our bottleneck already contains ConvNeXt blocks, see `UNDERSTANDING.md` §3.3) that maps the big
frozen `e_t` down to the abstract `c_t`.

```
data  ->  [FROZEN pretrained ViT  E]  ->  e_t   ->  [TRAINABLE bottleneck/CNN  B]  ->  c_t
          (never updated)                          (the only thing learning the representation)
```

### 3.3 Why this is a good idea

- **The hard part is already done.** Learning to see motion and objects from raw pixels is the
  expensive, data-hungry part. A pretrained video model (V-JEPA / V-JEPA 2) has already done it on
  far more data than we have. We inherit that for free.
- **It isolates the experiment.** Our actual research question is: *does the two-level coarse→fine
  predictive hierarchy work?* If the encoder is also training, a failure is ambiguous — bad encoder
  or bad hierarchy? Freezing the encoder removes that confound. Now only the bottleneck and the flow
  predictors are learning, so any signal is about *them*.
- **It is cheaper and more stable.** No encoder gradients, no encoder optimizer state, far fewer
  trainable parameters, and no "encoder collapses to a trivial solution" failure mode (a frozen
  network cannot collapse).

### 3.4 The notation `e_t = E(x_{≤t})`

`x_{≤t}` means "all the frames up to and including time `t`" — i.e. the context clip. So `e_t` is the
encoder's representation of the whole context clip, not a single frame. This matches what we already
do (encode the 4-frame context). The `≤t` is just precise notation for "the clip ending at `t`."

### 3.5 "from some sort of world model architecture like DINO or V-JEPA"

The supervisor is constraining *which* pretrained encoder. Not any ViT — one from a
**self-supervised / world-model** lineage, because those produce features tuned for *prediction and
physical understanding* rather than for classification labels. DINO/DINOv2 (image) and V-JEPA /
V-JEPA 2 (video) are the named families. The full comparison and the recommended pick live in
[`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md). The short version: V-JEPA 2 is the
natural fit because it is **video-native**, trained with a **JEPA** objective (the same family as
our design), uses the **same patch size (16)** we use, and is **state-of-the-art on Something-
Something V2 — which is literally our dataset.**

---

## 4. "Compute the EMA/stop-grad target c⁺_{t+k} = B_EMA(E(x_{≤t+k}))"

This is the target the flow predictor is trained to produce. Three things to unpack: the **target**,
the **EMA**, and the **stop-grad**.

### 4.1 What the target is

`c⁺_{t+k}` (read: "c-plus at t+k") is the **abstract latent of the future clip** — the clip that
ends `k` frames later, `x_{≤t+k}`. We run that future clip through the **same frozen encoder** `E`,
then through an **EMA copy of the bottleneck** `B_EMA`, to get the future abstract state we want to
predict.

Note the difference from `BRIEF_V0_1.md`: there the target was the encoding of a **single future
frame** `y`. Here the target is the encoding of the **whole clip up to `t+k`**. This is cleaner (it
matches how the encoder was pretrained — on clips, not single frames) and it generalizes naturally
to multiple horizons `k`.

### 4.2 What EMA is and why only `B` has one now

**EMA = Exponential Moving Average.** The "target" copy of a module is not trained by gradients; it
is a slowly-trailing average of the trained ("online") copy:

```
theta_EMA  <-  m * theta_EMA  +  (1 - m) * theta_online
```

with `m` close to 1 (e.g. 0.996 → 0.9999 over training). It lags behind the online module, changing
only gradually.

**Why have a lagging copy at all?** Because the predictor is chasing a target that is itself
produced by a network we are training. If the target moved at full speed every step, the predictor
would be aiming at a jittering bullseye, and the whole system could **collapse** (the easiest way to
make "predict your own output" succeed is for everything to become constant). A slow EMA target is
stable enough to be a meaningful thing to predict, and the well-known lag is what prevents the
trivial collapse. This is standard I-JEPA / V-JEPA practice.

**The change from before:** previously we EMA'd *both* the encoder (`E → E_bar`) and the bottleneck
(`B → B_bar`). Now the encoder is **frozen**, so it is *identical* on the online and target sides —
there is nothing to average. Only the **bottleneck** is being trained, so only the bottleneck needs
an EMA copy: `B_EMA`. The encoder `E` is shared, frozen, byte-for-byte the same on both branches.

```
ONLINE branch:  x_{≤t}    -> [frozen E] -> e_t         -> [trainable B]  -> c_t
TARGET branch:  x_{≤t+k}  -> [frozen E] -> e_{t+k}     -> [EMA B_EMA]    -> c⁺_{t+k}   (stop-grad)
                                  ^ same frozen weights both sides
```

### 4.3 What stop-grad means

**Stop-gradient** ("stopgrad", `.detach()`) means: we use the target value in the loss, but we do
**not** let gradients flow back into it. The predictor learns to hit the target; the target does not
bend toward the predictor. If we allowed gradients into the target, the cheapest way to reduce the
loss would be to make the target trivial — collapse again. Stop-grad on the target is the second
half of the anti-collapse safeguard (the EMA lag being the first half). `c⁺_{t+k}` is **always**
stop-grad.

---

## 5. The flow-matching mechanics, symbol by symbol

The supervisor gives the exact training recipe:

> Sample `z_0 ∼ N(0,I)`, `τ ∼ U(0,1)`, form `z_τ = (1−τ)z_0 + τ c⁺_{t+k}`, and train
> `v_θ(z_τ, τ, c_t, h_k)` to match `v_target = c⁺_{t+k} − z_0` using MSE:
> `L_flow = ||v_θ − (c⁺_{t+k} − z_0)||²`.

Let's decode each symbol and *why* it is what it is.

### 5.1 The straight-line path from noise to target

- **`z_0 ∼ N(0,I)`** — `z_0` is a fresh sample of standard Gaussian noise (mean 0, identity
  covariance), the same shape as `c_t`. This is the "speck of noise" we will transport to the
  future state. A different `z_0` each step is what lets the model represent multiple futures.
- **`c⁺_{t+k}`** — the (stop-grad, EMA) future abstract latent from §4. This is the destination.
- **`τ ∼ U(0,1)`** — `τ` (tau) is a random time between 0 and 1, drawn uniformly. `τ = 0` means
  "pure noise"; `τ = 1` means "the target." We sample a random point along the path each step.
- **`z_τ = (1−τ)z_0 + τ c⁺_{t+k}`** — this is the straight-line interpolation between noise (`z_0`)
  and target (`c⁺_{t+k}`). At `τ=0` it equals `z_0`; at `τ=1` it equals `c⁺_{t+k}`; in between it is
  a blend. `z_τ` is the point on the path the network will see this step. This straight line is
  exactly what makes it *rectified* flow.

### 5.2 The velocity target

- **`v_target = c⁺_{t+k} − z_0`** — this is the **velocity** of that straight-line path. Think of it
  as "how fast and in which direction you must move to get from the noise to the target in one unit
  of time." For a straight line traversed at constant speed, the velocity is just
  `(end − start) = c⁺_{t+k} − z_0`, and crucially it is the **same at every point on the line** (it
  does not depend on `τ`). That constancy is the whole convenience of rectified flow: the network
  only has to learn one direction per (noise, target) pair, not a curve.

### 5.3 The network and the loss

- **`v_θ(z_τ, τ, c_t, h_k)`** — the predictor (parameters `θ`). It is given:
  - `z_τ` — where it is on the path,
  - `τ` — how far along it is,
  - `c_t` — the present (the condition that steers the prediction),
  - `h_k` — the horizon embedding (which future, see §6).
  It outputs its **estimate of the velocity** at that point.
- **`L_flow = ||v_θ − (c⁺_{t+k} − z_0)||²`** — mean-squared error between the predicted velocity and
  the true velocity. That's it. Minimizing this over many random `(z_0, τ)` samples teaches the
  network the entire velocity field.

### 5.4 How you actually *predict* a future at inference

At training time you only ever learn the velocity. To get a concrete future abstract state at
inference, you start at noise (`τ=0`) and **integrate** the learned velocity field forward to `τ=1`
(a few small ODE steps; our design uses a 4-step Heun integrator). The endpoint is `c_hat`, a sampled
future abstract latent. (For a perfectly straight constant-velocity flow you could even do it in one
step: `c_hat = z_τ + (1−τ)·v_θ`. Our brief uses that one-step form during training for the
conditioning path, and multi-step Heun at inference for quality. See `UNDERSTANDING.md` §14 #7–#8.)

---

## 6. Multi-horizon prediction

### 6.1 What it is

Instead of always predicting "the next state," we predict the state **`k` frames into the future**
for several values of `k`. The supervisor specifies:

- **`k ∈ {4, 8, 16, 32}` frames** — four horizons, from near-future (4) to far-future (32).
- **Sampling probabilities `{0.30, 0.30, 0.25, 0.15}`** — for each training sample, we *randomly
  pick* one horizon `k` with these probabilities. Near horizons are sampled a bit more often than
  far ones (the far future is harder and noisier, so we weight it slightly less).

### 6.2 The horizon embedding `h_k`

We don't train four separate predictors. We train **one shared predictor** and *tell* it which
horizon it is aiming at by feeding in a **learned horizon embedding `h_k`** — a small vector, one per
horizon value, learned during training (like a positional embedding, but for "how far ahead").
`h_k` enters the predictor alongside `c_t`. The benefit: the horizons share almost all parameters
and reinforce each other (learning to predict 8 ahead helps predicting 4 and 16), while `h_k` lets
the single network specialize its output per horizon.

### 6.3 Why multi-horizon matters

A model that can only predict one step ahead is barely a world model. Forcing the *same* abstract
state and the *same* predictor to support 4-, 8-, 16-, and 32-frame predictions pressures `c_t` to
encode genuinely **temporally-extended, future-relevant structure** — the *dynamics*, not just the
next instant. It is a much stronger test (and teacher) of the hierarchy.

### 6.4 Scope decision: this is **Phase 4**, deferred

Per the project decision (see
[`ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md`](ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md)),
**multi-horizon is NOT part of the immediate work.** The immediate work is (1) swap in the frozen
encoder and (2) add the new metrics. Multi-horizon prediction becomes a new **Phase 4** after the
existing three phases. Until then, the predictor is single-horizon (effectively `k` fixed to the
one-step target), and `h_k` is documented but not yet wired in. This document explains it now so the
whole team understands where the architecture is heading.

---

## 7. Collapse prevention: variance floor instead of SIGReg

### 7.1 What "collapse" means

Collapse is the degenerate failure where the latent stops carrying information — e.g. `c_t` becomes
the **same constant vector** regardless of input. A constant target is trivially predictable (the
predictor just outputs the constant), so the loss looks great while the model has learned nothing.
Every JEPA-style design must actively prevent this.

### 7.2 What we used before (SIGReg) and why we're dropping it

`BRIEF_V0_1.md` used **SIGReg** (Sketched Isotropic Gaussian Regularization, from the LeJEPA line):
a loss that pushes the latent distribution toward an isotropic Gaussian using many random
projections and a characteristic-function statistic. It is principled but heavier — random
projection counts, integration knots, applied to both `e_t` and `c_t`.

Two reasons to drop it now:
1. **`e_t` is frozen.** SIGReg on `e_t` is pointless — you cannot reshape the output distribution of
   a frozen encoder with a gradient. So half of the old SIGReg usage is dead on arrival.
2. **The supervisor wants minimal machinery.** The only thing that can now collapse is `c_t` (the
   bottleneck's output), and the flow objective plus the EMA/stop-grad target already do most of the
   anti-collapse work. So we keep the *simplest possible* explicit safeguard and let the temporal
   flow objective learn the semantics.

> The supervisor is explicit: **"no SIGReg, no full VICReg, no covariance loss initially."** Do not
> add them back without approval.

### 7.3 The variance floor, term by term

> `L_var = (1/d) Σ_j max(0, 1.0 − Std(c_j))`, total `L = L_flow + 0.1·L_var`.

How to compute it:
1. **Flatten `c_t` across slots/features per batch.** `c_t` has shape `(B, N_c, D_c)` (batch ×
   abstract tokens × abstract dim). Flatten so you have a 2-D matrix of "rows = samples, columns =
   `d` feature dimensions." (Treat the per-slot features as the `d` dimensions; pool/flatten exactly
   as the implementation specifies — the key is you end with a per-dimension view across the batch.)
2. **Per-dimension std.** For each of the `d` columns `j`, compute the standard deviation `Std(c_j)`
   across the batch.
3. **Hinge at 1.0.** For each dimension, take `max(0, 1.0 − Std(c_j))`. If a dimension's std is
   already ≥ 1.0, this term is 0 (no penalty — that dimension is healthy and varied). If its std is
   small (the dimension is nearly constant — collapsing), the term is large (a penalty that pushes
   it back up).
4. **Average over dimensions** (`(1/d) Σ_j`) to get one scalar.
5. **Weight by 0.1 and add to the flow loss.** `L = L_flow + 0.1·L_var`.

### 7.4 The critical mindset

> "The variance floor is only to prevent constant `c_t`; the temporal flow objective should learn
> the actual semantics/dynamics."

This is the most important sentence in the whole update. The variance floor is a **guardrail, not a
teacher.** It only stops dimensions from flatlining. It does **not** make `c_t` meaningful — only the
flow-matching prediction objective does that. Do not be tempted to crank up the `0.1` weight or add
covariance terms to "improve" the representation; that is not its job, and over-weighting it just
inflates variance without adding information.

---

## 8. The metrics we must log (at minimum)

The supervisor names three monitors. These are how we *see* whether `c_t` is healthy. All three are
on `c_t` specifically.

### 8.1 Variance of `c_t`

- **What:** the spread of `c_t` values — typically the mean (and per-dimension distribution) of the
  per-dimension standard deviations from §7.3.
- **Why:** it is the direct readout of collapse. If variance trends toward zero, `c_t` is becoming
  constant — the model is collapsing — stop the run.
- **Healthy looks like:** per-dimension stds comfortably away from zero (the variance-floor hinge at
  1.0 is the target neighborhood), no large fraction of dimensions near zero.

### 8.2 Cosine similarity of `c_t` from one video to another

- **What:** take `c_t` for two *different* videos, flatten each to a vector, and compute their cosine
  similarity (the cosine of the angle between them: `(a·b)/(|a||b|)`, ranging −1 to 1). Average over
  many random pairs in the batch.
- **Why:** this catches a subtler collapse than variance alone. The latent could have non-zero
  variance but still map **every** video to nearly the **same direction** — different magnitudes,
  same content. If the average cross-video cosine similarity is near 1.0, all videos look alike to
  `c_t`: the representation is not discriminative. We want clearly *different* videos to have clearly
  *different* `c_t`, i.e. **low** average cross-video cosine similarity.
- **Healthy looks like:** average pairwise cosine similarity well below 1.0 (distinct videos →
  distinct abstract states). A value drifting toward 1.0 is a collapse warning even if variance is
  fine.

### 8.3 Effective rank of `c_t`

- **What:** "effective rank" measures how many dimensions the latent *actually* uses. Compute the
  covariance matrix of `c_t` across the batch, take its eigenvalues, normalize them to sum to 1, and
  compute the entropy-based effective rank (`exp(−Σ p_i log p_i)`). A latent that genuinely spreads
  information across many dimensions has a high effective rank; one that secretly lives on a tiny
  subspace has a low one.
- **Why:** it catches **dimensional collapse** — the case where the latent uses only a handful of
  directions and wastes the rest. Variance and cosine can both look acceptable while the latent has
  quietly collapsed onto a low-dimensional subspace; effective rank exposes that.
- **Healthy looks like:** effective rank a large fraction of `D_c` (our existing brief used "> 60 of
  256 = healthy, < 20 = concerning, < 5 = hard stop" for `c_t`; we carry those thresholds forward).

> **All three together.** Variance catches "everything went constant." Cosine catches "everything
> points the same way." Effective rank catches "everything lives in a tiny subspace." You need all
> three because each misses what the others catch.

---

## 9. The new pipeline, end to end

Putting it all together, the updated single-horizon pipeline (Phase 1 target state) is:

```
                      ┌─────────────── FROZEN (pretrained, never updated) ───────────────┐
context clip x_{≤t} ──►  pretrained ViT  E  ──► e_t ─┐
                      └────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                          [TRAINABLE bottleneck/CNN  B] ──► c_t
                                                     │
                                                     │  (condition)
                                                     ▼
   z_0 ~ N(0,I), τ ~ U(0,1) ──► z_τ ──► [TRAINABLE flow predictor v_θ(z_τ, τ, c_t, h_k)] ──► v̂
                                                     │
                                                     ▼
                                  L_flow = || v̂ − (c⁺_{t+k} − z_0) ||²
                                                     +
                                  0.1 · L_var   (variance floor on c_t only)

TARGET (stop-grad):
future clip x_{≤t+k} ──► [same FROZEN E] ──► e_{t+k} ──► [EMA bottleneck B_EMA] ──► c⁺_{t+k}  (detached)

MONITOR every N steps:  variance(c_t) · cross-video cosine(c_t) · effective-rank(c_t)
```

For Phase 1 right now, `k` is the single one-step horizon and `h_k` is not yet wired (multi-horizon
is Phase 4). The fine flow `F_e`, the frame generator `D`, and Stages 2–4 are unchanged by this
update except that they now consume the frozen-encoder `e_t` (different dimensionality) — see
[`ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md`](ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md).

---

## 10. What changed vs. what already matched

| Topic | `BRIEF_V0_1.md` (original) | Supervisor update (`v0.2`) | New? |
|---|---|---|---|
| Coarse predictor type | Conditional rectified-flow over `c_t` | Conditional rectified-flow over `c_t` | **No change** — already matched |
| Encoder `E` | VideoViT-S trained **from scratch** | **Frozen pretrained** ViT (DINO/V-JEPA family) | **Major change** |
| Encoder dim `D_e` | 384 | Set by the pretrained model (e.g. 768 / 1024) | **Major change** |
| Target encoder | EMA copy `E_bar` (trained) | Same frozen `E`, shared both branches | **Major change** |
| EMA scope | Both `E_bar` and `B_bar` | **Only** `B_EMA` (encoder frozen) | **Major change** |
| Prediction target | Encode single future frame `y` | Encode future clip `x_{≤t+k}` → `c⁺_{t+k}` | **Change** |
| Horizons | One step (t+1) | Multi-horizon `k ∈ {4,8,16,32}` + `h_k` | **New (deferred to Phase 4)** |
| Collapse prevention | SIGReg on `e_t` and `c_t` | **Variance floor on `c_t` only**; no SIGReg/VICReg/covariance | **Change** |
| Total loss | `L_c + λ_fine L_e + λ_e SIGReg(e) + λ_c SIGReg(c)` | `L_flow + 0.1·L_var` (coarse stage) | **Change** |
| Required monitors | std, effective rank, baselines, shuffled-c, etc. | + cross-video cosine of `c_t` explicitly; variance + eff. rank of `c_t` | **Added** |
| Stop-grad on target | Always | Always | **No change** |

---

## 11. Open questions this document deliberately does not answer

These are resolved elsewhere (and a couple still need your input):

- **Which exact pretrained encoder, and at what size/resolution?** →
  [`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md) (with two recommended options and a
  follow-up question for you).
- **How does `e_t`'s new dimensionality cascade through the bottleneck, fine flow, and frame
  generator constants?** → the per-file diff in
  [`ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md`](ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md) and the rewrite
  of [`UNDERSTANDING.md`](UNDERSTANDING.md).
- **Do we keep tubelet dropout on a frozen encoder?** → flagged as a research/decision item in
  `FROZEN_ENCODER_RESEARCH.md` §"Cascade".
- **Exact `L_var` flatten convention (slots × features) in code.** → to be locked when code is
  written; the math here is the contract.
