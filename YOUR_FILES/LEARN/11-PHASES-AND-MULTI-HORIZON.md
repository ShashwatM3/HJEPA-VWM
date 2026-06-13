# 11 — The Roadmap: Phases 1–4 and How Multi-Horizon Actually Works

> **What you'll understand after this file:** what each phase adds and what
> question it answers, why the order is what it is, and — in full detail —
> the multi-horizon design your tech lead raised: the horizon embedding,
> the sampling scheme, and what changes (and doesn't) in the pipeline.

---

## 1. Why phases at all

The architecture has many parts that *could* be trained jointly: encoder,
bottleneck, coarse predictor, fine predictor, VAE, frame generator. Joint
training of everything is how you get a system whose failures cannot be
attributed. The project instead follows a strict rule: **each phase adds
one capability, has its own falsifiable acceptance gates, and only
unlocks the next phase by passing them.** A failure is then localized to
the newest component by construction. (Same logic as freezing the encoder
— scientific control, applied to the schedule instead of the
architecture.)

The dependency chain, and the question each phase answers:

| Phase | Adds | Question answered | Status |
|---|---|---|---|
| **1** | Bottleneck `B` + coarse flow `F_c` | *Can a compressed latent learn coarse dynamics at one horizon?* | implemented; Run 2 pending |
| **2** | Fine flow `F_e` (predicts future `e` conditioned on predicted `c`) | *Does the hierarchy work — does the abstract level steer the detailed level?* | spec'd |
| **3** | VAE `A` + frame generator `D` | *Can predicted latents be decoded into actual pixels?* | spec'd |
| **4** | Multi-horizon coarse prediction | *Can ONE predictor handle multiple time scales?* | spec'd, deferred |

## 2. Phase 2 in brief: the hierarchy test

`F_e` is a second flow-matching predictor, identical recipe (file 04), but
its target is the *detailed* future `e⁺` and its conditioning includes the
**predicted** abstract future `ĉ` from `F_c`. The decisive diagnostic is
the **shuffled-c test** (file 01 §4): condition `F_e` on the *wrong*
clip's `c` and require its loss to get much worse. Passing means the
coarse level genuinely informs the fine level — the "H" in HJEPA is real.
Failing means the hierarchy is decorative and the architecture needs
rethinking *before* any pixels are generated. Phase 2 is therefore the
make-or-break test of the project's central hypothesis.

## 3. Phase 3 in brief: pixels at last

A VAE (`A`) maps frames to a compact pixel-latent space; a frame generator
(`D`) — once more the same flow-matching recipe — generates future frames
in that VAE space, conditioned on the predicted `e`/`c`. Only here does
the model produce anything you can *watch*. Note the design consistency:
three predictors (coarse, fine, frame), one generative recipe. Every
conceptual tool you learned in file 04 amortizes across all three.

## 4. Phase 4: multi-horizon, properly explained

### 4a. The limitation it removes

Phase 1's predictor answers exactly one question: "what does the world
look like **4 frames** (≈ a third of a second) from now?" `horizon_k = 4`
is baked into the data pipeline — the target window always sits 4 frames
ahead. A world model with one fixed lookahead is like a chess engine that
can only think exactly one move ahead: useful, but not *planning*.
Different decisions need different time scales — "where will the hand be
in 0.3s?" vs "will the cup be off the table in 3s?"

### 4b. The naive fix and why it's rejected

Train four separate predictors, one per horizon. Cost: 4× the parameters,
4× the training, and — the deep objection — **nothing shared**: what the
model learns about 8-frame dynamics teaches the 16-frame predictor
nothing. Physical dynamics compose across time scales; separate heads
can't exploit that.

### 4c. The actual design: one predictor + a horizon embedding

The supervisor's design (spec: `AGENT_FILES/PHASES/PHASE_4.md`) keeps
**one shared `F_c`** and tells it which horizon it's aiming at:

```
v̂ = F_c(z_τ, τ, c_t, h_k)
```

- **Horizon set:** `k ∈ {4, 8, 16, 32}` frames.
- **`h_k`** is a learned embedding table — literally 4 vectors of dim 256
  (`horizon_embed_dim = D_c`), one per horizon value, initialized small
  and shaped entirely by training.
- **Injection:** one of two wirings (the spec allows either, demands the
  choice be documented):
  1. **Additive to the time embedding** — `h_k` joins τ's embedding at the
     adaLN input, so the horizon *modulates every block* the same way flow
     time does ("process this prediction in 16-frame mode").
  2. **Extra conditioning token** — `h_k` becomes a 65th token in the
     sequence, readable by attention like the `c_t` tokens.

  Wiring 1 is the more natural fit conceptually: horizon, like τ, is a
  *global mode* of the computation rather than a piece of content — and
  adaLN is precisely the machinery for global modes (file 04 §3b).

This is the same design pattern you've now seen twice — learned queries
(file 03), null-condition token (file 04): **when a discrete choice must
influence a network, give each choice a learned vector and let training
decide what it means.** Nobody hand-designs what "16-frames-ahead mode"
means; `h_16` becomes whatever vector makes 16-frame predictions work.

### 4d. The training step — diff against Phase 1

Per sample (not per batch — each batch mixes horizons):

```
k  ~ Categorical({4, 8, 16, 32}, p = {0.30, 0.30, 0.25, 0.15})
target clip = the window ending at t + k        ← data pipeline change
c⁺ = B_EMA(E(x_{≤t+k}))                          ← same machinery, k-dependent window
v̂  = F_c(z_τ, τ, c_t, h_k)                      ← new argument
L  = ‖v̂ − (c⁺ − ε)‖² + 0.1·L_var(c_t)            ← unchanged
```

What does **not** change is most of the lesson:

- Encoder: frozen, untouched (Phase 4 explicitly forbids encoder changes).
- Bottleneck and `c_t`: unchanged — one `c_t` per sample regardless of k;
  the variance floor still acts on it identically.
- EMA, stop-grad, flow-matching math: identical.
- Tubelet geometry: identical — and this answers the question you asked
  once: multi-horizon does *not* change tokenization. "Predict further
  ahead" means *the target window slides further into the video* (the
  dataloader samples a clip ending at t+k instead of t+4); the future
  clip is still 8 frames → 1024 tubelet tokens → `B_EMA` → a 32×256
  target. Horizons change *which* future is encoded, never *how*.

The sampling probabilities lean toward near horizons (0.30/0.30/0.25/0.15)
— near-future prediction is the foundation skill and the more reliable
gradient signal; far horizons are harder and noisier (more accumulated
uncertainty), so they get proportionally less of the training budget.

Data-pipeline corner case worth noticing: `k = 32` at stride 2 needs a
span of `(8−1)·2 + 32 = 46` frames; many SSv2 clips are ~30–90 frames, so
short videos can't supply far horizons. The spec requires deterministic
skip/pad with *logged counts* — silent data filtering is how datasets
quietly become unrepresentative (same MLOps reflex as the not-[-1,1]
assert in file 08).

### 4e. How Phase 4 is judged

Diagnostics become **per-horizon** — `L_flow` and the copy/batch-mean
baselines logged separately for each k. The gates encode physical common
sense:

1. Every horizon trains (loss decreases, no NaN).
2. `F_c` beats copy & batch-mean *at every k* — a far-horizon predictor
   that can't beat "nothing changes" hasn't learned long-range dynamics.
3. **Monotonicity:** far horizons (16, 32) must show *higher* loss than
   near (4, 8) — while still beating their baselines. If 32-frame loss
   matched 4-frame loss, the model would likely be predicting something
   degenerate rather than genuinely harder futures. (A metric where
   "doing worse" is *required* — diagnostics testing for the right shape
   of difficulty, not just low numbers.)
4. `c_t` health thresholds unchanged.
5. One shared predictor — no per-horizon weights anywhere.

An optional diagnostic measures **target drift vs k**: how far `c⁺_{t+k}`
sits from `c_t` as k grows. It validates the premise — if 32-frame
futures weren't actually farther from the present in latent space, the
"far horizons are harder" assumption would be wrong for this data.

### 4f. Why Phase 4 is deferred (and what it would inherit today)

Multi-horizon multiplies the *evaluation* surface (everything ×4) and
slightly complicates data sampling — all worthless if single-horizon
prediction doesn't work, and confusing to debug if latent health is
already questionable. It extends `F_c` — so it inherits whatever Phase 1
produces. Concretely: if `c_effective_rank` is still ~5 when Phase 4
starts, all four horizons will be predicting within that same impoverished
5-dim subspace, and far-horizon gates will be testing dynamics the latent
may not even represent. Fixing latent richness *first* is not
perfectionism; it's sequencing.

## 5. Questions to test yourself

1. Why one shared predictor instead of four? *(Parameter/compute economy,
   and cross-horizon transfer — dynamics knowledge composes across time
   scales.)*
2. What exactly is `h_k`, and what determines its meaning? *(A learned
   row in a 4-entry embedding table; meaning emerges from training
   pressure — nothing is hand-designed.)*
3. Does k=32 change the number of tokens the encoder produces? *(No —
   the future clip is still 8 frames/1024 tokens; only the window's
   position in the video changes.)*
4. Why must far-horizon loss be HIGHER for the phase to pass? *(Equal
   loss across horizons would signal degenerate prediction; genuine
   far-future prediction is necessarily harder.)*
5. Why are near horizons sampled more often? *(More reliable signal,
   foundation skill; far horizons are noisier and get a smaller share of
   the budget.)*
6. Your tech lead asks "can we add k=64?" — what breaks first? *(Data:
   span = (8−1)·2 + 64 = 78 frames; most SSv2 clips are too short, so
   k=64 would train on a thin, biased subset of long videos.)*
