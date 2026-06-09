# BRIEF_V0_2.md — Hierarchical JEPA-Flow Video World Model (frozen-encoder update)

> **What this file is.** The **working** architecture brief. It is a duplicate of
> [`BRIEF_V0_1.md`](BRIEF_V0_1.md) (the faithful transcription of the original PDF) **edited in
> place** to fold in the supervisor's update: a frozen pretrained ViT encoder, an EMA on the
> bottleneck only, a clip-level flow-matching target, a variance-floor collapse term replacing
> SIGReg, and multi-horizon prediction (deferred to Phase 4).
>
> **Precedence.** This file (with [`SUPERVISOR_FEEDBACK_EXPLAINED.md`](SUPERVISOR_FEEDBACK_EXPLAINED.md))
> supersedes `BRIEF_V0_1.md` wherever they differ. Numerical constants are mirrored in
> [`UNDERSTANDING.md`](UNDERSTANDING.md) §2.6 (single source of truth). For the conceptual
> walkthrough read the feedback explainer; for the encoder choice see
> [`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md).

---

## Changes vs v0.1 (read first)

| # | Change | v0.1 | v0.2 |
|---|---|---|---|
| 1 | Encoder `E` | VideoViT-S trained from scratch (dim 384) | **Frozen pretrained V-JEPA 2 ViT-L/16** (dim 1024), shared by both branches |
| 2 | Target encoder | EMA copy `E_bar` | **Same frozen `E`** (no `E_bar`) |
| 3 | EMA scope | `E_bar` + `B_bar` | **`B_EMA` only** |
| 4 | Prediction target | encode single future frame `y` | **`c⁺_{t+k} = B_EMA(E(x_{≤t+k}))`** (clip-level) |
| 5 | Collapse prevention | SIGReg on `e_t` and `c_t` | **Variance floor on `c_t` only** (no SIGReg/VICReg/covariance) |
| 6 | Coarse-stage loss | `L_c + λ_e SIGReg(e) + λ_c SIGReg(c)` | **`L = L_flow + 0.1·L_var`** |
| 7 | Horizons | t+1 | **multi-horizon `k∈{4,8,16,32}` + `h_k`** — **deferred to Phase 4** |
| 8 | Context | 4 frames @128 | **8 frames @256** (tubelet-2 → 4 temporal tokens) |
| 9 | Tubelet dropout | 40% on context | **Removed** from encoder input (frozen model) |
| 10 | Required monitors | std, eff-rank, baselines | **+ cross-video cosine of `c_t`** (variance + eff-rank of `c_t` explicit) |

**Unchanged:** the two-level hierarchy, rectified-flow predictors, stop-grad on all targets, the
bypass-test mindset (shuffled-c, decoder-dependency), `c_t` low-bandwidth vs `e_t`, and the
gradient-routing rules (`L_e ↛ F_c`, frame loss ↛ latent stack).

---

## Plain-English architecture brief for coder review

**One-sentence proposition:** train a video world model that first predicts the next abstract
latent state, then predicts the next detailed latent state, and only after that trains a separate
generator to render the next frame.

The goal is not to build a normal video diffusion model. The goal is to build a **predictive
latent hierarchy**: the compressed state must carry the future-relevant structure, while the
detailed latent carries texture and local visual information.

**Key rule:** the frame decoder is not allowed to teach the encoder during main training. The
encoder is **pretrained and frozen**; the *bottleneck* is taught by future latent prediction against
a stop-gradient EMA target branch.

---

## 1. The architecture in plain English

At time `t`, the model receives a short context clip (the last **8** video frames). A **frozen
pretrained ViT** turns that clip into detailed tokens called `e_t`. A trainable bottleneck module
then compresses `e_t` into a much smaller abstract state called `c_t`.

For a target horizon `k`, the **same frozen encoder** looks at the future clip `x_{≤t+k}` and a
**slow-moving EMA copy of the bottleneck** produces the target abstract latent `c⁺_{t+k}`. This
target is stop-gradient: the model predicts it, but does not backpropagate into it.

A coarse flow model learns to generate `c_hat` from `c_t`. Then a fine flow model learns to generate
`e_hat` from `e_t` plus a coarse future state. Finally, after latent training works, a
cross-attention frame generator renders the predicted next frame from `e_hat`.

```
Data path
---------
context clip x_{<=t}   -> [FROZEN pretrained ViT E] -> detailed latent e_t
e_t                    -> [TRAINABLE bottleneck B]  -> abstract latent c_t
future clip x_{<=t+k}  -> [SAME FROZEN E]           -> e_{t+k}
e_{t+k}                -> [EMA bottleneck B_EMA]     -> stopgrad c_plus (= c⁺_{t+k})
```

```
Prediction path
---------------
c_t -> coarse flow F_c -> c_hat
(e_t, c_plus) early -> fine flow F_e -> e_hat   [teacher-forced]
(e_t, stopgrad(c_hat)) late -> fine flow F_e -> e_hat   [real rollout]
stopgrad(e_hat) -> frame generator -> x_hat
```

> **OPEN DESIGN DECISION (Phase 2/3 target geometry).** The supervisor's update specifies the
> **coarse/abstract** path only. With a frozen *video* encoder, the natural detailed target for the
> fine flow is the future **clip** latent `e_{t+k} = E(x_{≤t+k})` (same geometry as `e_t`), and the
> frame generator would render the last frame of that future clip (frame `t+k`). This is the
> recommended default but should be confirmed with the supervisor before Phase 2 code. See
> [`ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md`](ARCHITECTURE_CHANGES_AND_PHASE_PLAN.md) §4.

---

## 2. Recommended v0 dimensions

| Component | Recommended v0.2 | Reason |
|---|---|---|
| Input | **8 context frames at 256 × 256**; predict horizon `k` (single horizon in Phases 1–3) | Short clips give velocity; 8 frames → 4 temporal tokens for real motion in a tubelet-2 video encoder. |
| Patch / tubelet size | **16 × 16, tubelet 2** (set by the encoder) | V-JEPA 2 native geometry: (8/2) × (256/16) × (256/16) = 4 × 16 × 16 = 1024 context tokens. |
| Frozen encoder E | **Frozen V-JEPA 2 ViT-L/16** (depth 24, dim **1024**, 16 heads, native res 256), `requires_grad=False` | Pretrained video world model; SSv2 SOTA family; isolates the hierarchy as the only thing learned. |
| Detailed latent e_t | **1024 tokens × 1024 dim** (context); future-clip target same geometry | Set by the frozen encoder; rich appearance + motion. |
| Bottleneck B | ConvNeXt-style token mixer + 32 learned query slots, input proj **1024 → 256** | Forces compression from detailed perceptual state into abstract state. **Only trainable encoder-side module.** |
| Abstract latent c_t | 32 tokens × 256 dim | Tight bottleneck; low bandwidth vs `e_t` to prevent copying. |
| Coarse flow F_c | 6 transformer blocks, dim 256, 8 heads (+ horizon embed `h_k` in Phase 4) | Predicts future abstract state `c⁺_{t+k}` from current `c_t`. |
| Fine flow F_e | 8 transformer blocks, dim 384, 8 heads; cross-attn memory width follows `D_e=1024` | Predicts future detailed latent from `e_t` and future coarse state. |
| Frame generator D | 12-block cross-attention flow transformer in VAE latent space | Render plausible frames after the latent world model works. |

---

## 3. What teaches the encoder?

**Nothing trains the encoder — it is frozen.** What is *taught* is the **bottleneck** `B` and the
flow predictors. The bottleneck must produce `c_t` (and the flows must produce `c_hat` / `e_hat`)
that recover the future target latents.

The target abstract latent comes from the **EMA bottleneck** `B_EMA` applied to the **frozen
encoder's** output on the future clip. Because the encoder is frozen and shared, only the bottleneck
needs an EMA copy. The EMA lag + stop-grad prevents the target from chasing the predictor and
prevents collapse.

```
EMA target update (bottleneck only)
-----------------------------------
B_EMA <- m * B_EMA + (1 - m) * B

Recommended m schedule:
start 0.996, cosine-ramp toward 0.9999

(No E_bar: the encoder is frozen and identical on both branches.)
```

---

## 4. Losses

### 4.1 Coarse JEPA-flow loss (now the Phase-1 mainline)

The coarse flow learns to transform Gaussian noise into the future abstract latent `c⁺_{t+k}`,
conditioned on `c_t` (and, in Phase 4, on the horizon embedding `h_k`). Conditional rectified flow.

```
Sample z_0 ~ N(0, I), tau ~ U(0, 1)
z_tau    = (1 - tau) * z_0 + tau * c_plus           # c_plus = c⁺_{t+k}, stop-grad
v_target = c_plus - z_0
L_flow   = mean_square( v_theta(z_tau, tau, c_t [, h_k]) - v_target )

Gradients from L_flow update:        B, F_c   (encoder E is FROZEN)
Gradients from L_flow do NOT update: E, B_EMA, c_plus
```

### 4.2 Fine JEPA-flow loss

Unchanged in form; consumes the wider `e_t` (dim 1024). Early training uses teacher-forced `c_plus`,
late training uses detached `c_hat`. **`L_e` must not update `F_c` through `c_hat`.** (See the Phase
2/3 target-geometry open decision in §1.)

```
c_condition = c_plus            (Stage 2)   |   stopgrad(c_hat)   (Stage 3)
z_e = (1 - tau) * eps_e + tau * e_plus
u_e = e_plus - eps_e
L_e = mean_square( F_e(z_e, tau, e_t, c_condition) - u_e )
```

### 4.3 Collapse prevention — variance floor (replaces SIGReg)

**Do not use SIGReg, full VICReg, or a covariance loss.** Collapse prevention is minimal: a
variance floor on `c_t` only (the encoder is frozen and cannot collapse; only the bottleneck output
can).

```
Flatten c_t across slots/features per batch; compute per-dimension std.
L_var = (1 / d) * sum_j max(0, 1.0 - Std(c_j))

Total (coarse stage):
L = L_flow + 0.1 * L_var
```

The variance floor only prevents constant `c_t`. The temporal flow objective is what learns the
semantics/dynamics. Do not over-weight `L_var`.

### 4.4 Frame generator loss

Unchanged: after latent training works, freeze the world model and train the frame generator in a
frozen image-VAE latent space.

```
a_y = A(true_future_frame)         # frozen image VAE
z_x = (1 - tau) * eps_x + tau * a_y
u_x = a_y - eps_x
L_frame = mean_square( D(z_x, tau, stopgrad(e_hat)) - u_x )

Gradients from L_frame update:        D only
Gradients from L_frame do NOT update: E, B, F_c, F_e
```

---

## 5. Stop-gradient rules

| Object | Stop-gradient? | Why |
|---|---|---|
| `e_plus` (= `e_{t+k}`), `c_plus` (= `c⁺_{t+k}`) | Always yes | Targets from the EMA / frozen branch. |
| Encoder `E` | **Frozen (no grad ever, both branches)** | Pretrained; never updated. |
| `B_EMA` | No backprop ever | Updates only through EMA. |
| `c_hat` when fed into F_e | Yes during predicted-coarse training | Prevents `L_e` from corrupting the coarse predictor. |
| `e_hat` when fed into D | Yes during frame-generator training | Frame generation must not rewrite the world model. |
| Bottleneck `B`, `c_t` on conditioning path | No stopgrad during latent training | It must learn to preserve predictive information. |

---

## 6. Training schedule

| Stage | Train | Loss | Pass condition |
|---|---|---|---|
| 0. Setup | Init `B_EMA = B`; load + freeze `E` | None | Loader, shapes, EMA update, mixed precision stable; frozen-encoder forward works. |
| 1. Coarse dynamics | **B, F_c** (E frozen) | `L_flow + 0.1·L_var` | `c_t` noncollapsed (variance / cosine / eff-rank); `F_c` beats copy & batch-mean baselines. |
| 2. Fine teacher forcing | B, F_c, F_e | `L_flow + L_e + 0.1·L_var` | `F_e` works with true stopped `c_plus`. |
| 3. Predicted-coarse fine | B, F_c, F_e | `L_flow + L_e + 0.1·L_var` | Ramp `c_condition` to `stopgrad(c_hat)`; shuffled `c_hat` hurts. |
| 4. Frame generator | D only; freeze E, B, F_c, F_e | `L_frame` | Frames depend strongly on `e_hat`; shuffled `e_hat` fails. |
| (Phase 4) Multi-horizon | B, F_c (+ `h_k`) | `L_flow + 0.1·L_var`, multi-`k` | Predicts across `k∈{4,8,16,32}`; per-horizon health. |

> Multi-horizon is implemented as **Phase 4** after the existing three phases (see `PHASE_4.md`).

---

## 7. Optimizer and runtime defaults

| Setting | Recommended default |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05 |
| Gradient clipping | Global norm 1.0 |
| Precision | bfloat16 AMP (encoder forward under `no_grad`) |
| Batch | Global batch about 64 clips, gradient accumulation if needed |
| Learning rate: **encoder** | **none (frozen)** |
| Learning rate: bottleneck | 2e-4 |
| Learning rate: flows | 4e-4 for F_c and F_e |
| Learning rate: frame generator | 2e-4 |
| Schedule | 10k warmup, cosine decay |
| Tubelet dropout | **Removed** from the frozen encoder input (optional light masking inside the bottleneck) |
| Condition dropout | About 10 percent for flow conditioning robustness |
| Input normalization | **The encoder's expected normalization** (V-JEPA processor), separate from the VAE's [−1,1] used in Stage 4 |

---

## 8. Dataset recommendation

Use Something-Something V2 for v0. It is action and object-interaction heavy, so temporal order
matters — and V-JEPA 2 is **state-of-the-art on SSv2**, making it an especially good frozen encoder.

Practical setting: train at **256 × 256** with 8-frame context; predict a single horizon first, then
add multi-horizon (Phase 4). Do not begin with long rollouts. First prove the hierarchy works.

---

## 9. Runtime tests and early-stop gates

| Test | What to measure | Stop or investigate if |
|---|---|---|
| **Variance of `c_t`** | Per-dimension std of `c_t` | Many dimensions approach zero. |
| **Cross-video cosine of `c_t`** | Mean pairwise cosine similarity of `c_t` across different videos | Approaches 1.0 (all videos look alike). |
| **Effective rank of `c_t`** | Covariance effective rank of `c_t` | Collapses sharply or keeps falling. |
| Coarse baseline | `F_c` error vs copy `c_t` and batch-mean | `F_c` does not beat baselines after warmup. |
| Shuffled c test | `F_e` with real `c_hat` vs shuffled/zeroed | Shuffled performs almost as well as real. |
| Teacher vs predicted gap | `F_e` with `c_plus` vs `c_hat` | Predicted-coarse never approaches teacher-forced. |
| Gradient health | Grad norm, NaNs, loss spikes | Repeated spikes, NaNs, or exploding EMA drift. |
| Decoder dependency | D with true/predicted/shuffled `e_hat` | Shuffled `e_hat` still gives plausible frames. |

The first three (variance, cross-video cosine, effective rank of `c_t`) are the supervisor's
**minimum required** monitors.

---

## 10. What the coder should implement first

The first implementation should not include the frame generator. The first milestone is only the
latent model: **frozen encoder load**, trainable bottleneck, **EMA bottleneck branch**, coarse flow,
fine flow with teacher forcing, **variance floor**, and the collapse/bypass tests.

- **Milestone 1:** training script runs Stage 1 and proves `c_t` is noncollapsed (variance / cosine
  / eff-rank) and `F_c` beats copy/batch-mean baselines.
- **Milestone 2:** add `F_e` with teacher-forced `c_plus`; prove the detailed prediction depends on
  `c_plus`.
- **Milestone 3:** switch to `stopgrad(c_hat)`, prove shuffled `c_hat` hurts, then add the frame
  generator.
- **Later (Phase 4):** add multi-horizon prediction.

---

## 11. Non-negotiable design constraints

| Constraint | Reason |
|---|---|
| Encoder is **pretrained and frozen**; only the bottleneck/flows train. | Isolates the hierarchy as the experiment; removes encoder-collapse failure mode. |
| The target branch is always stop-gradient and EMA-updated (**bottleneck EMA**). | Otherwise the target moves too fast and the representation can collapse or chase itself. |
| `c_t` must be much lower bandwidth than `e_t`. | Otherwise the abstract state becomes a copy of the detailed latent. |
| `F_e` must be tested with shuffled/zeroed c. | The main test that the hierarchy is real, not decorative. |
| `L_e` must not backprop through `c_hat` into `F_c`. | Otherwise the fine loss turns c into a texture carrier. |
| The frame decoder is trained after latent learning and does not update the encoder/bottleneck. | Otherwise it degenerates into a normal video generator. |
| Collapse prevention is the **variance floor on `c_t` only** — no SIGReg/VICReg/covariance. | Supervisor directive; minimal machinery, let the flow objective learn semantics. |

---

## 12. Literature basis

| Idea used | Reference basis |
|---|---|
| Predicting representations instead of pixels | I-JEPA, arXiv:2301.08243. |
| Latent video prediction / frozen video world-model encoder | V-JEPA (ICLR 2024); **V-JEPA 2**, arXiv:2506.09985 (encoder we use). |
| Flow matching / rectified flow | Flow Matching for Generative Modeling, arXiv:2210.02747; Liu et al. rectified flow. |
| Latent-space generation & cross-attention conditioning | Latent Diffusion Models, arXiv:2112.10752. |
| adaLN-Zero time conditioning | DiT (Peebles & Xie, 2022). |
| Variance-floor collapse prevention | VICReg variance term (Bardes et al.) — *variance term only*, not the full method. |
| Dataset | Something-Something V2 (~220k human-object interaction videos). |
