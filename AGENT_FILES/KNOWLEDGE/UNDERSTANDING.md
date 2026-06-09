# UNDERSTANDING.md
## Hierarchical JEPA-Flow Video World Model — Comprehension Reference

> **⚠️ v0.2 UPDATE BANNER (read first).** This document has been updated in place to reflect the
> supervisor's flow-matching / frozen-encoder / multi-horizon update. The authoritative spec is now
> [`BRIEF_V0_2.md`](BRIEF_V0_2.md) (the edited brief), with the conceptual walkthrough in
> [`SUPERVISOR_FEEDBACK_EXPLAINED.md`](SUPERVISOR_FEEDBACK_EXPLAINED.md) and the encoder rationale in
> [`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md). Key changes folded in below:
> **(1)** encoder `E` is a **frozen pretrained V-JEPA 2 ViT-L/16** (dim `D_e=1024`), shared by both
> branches — not trained from scratch; **(2)** EMA is on the **bottleneck only** (`B_EMA`); there is
> no `E_bar`; **(3)** the target is clip-level `c⁺_{t+k} = B_EMA(E(x_{≤t+k}))`; **(4)** collapse
> prevention is a **variance floor on `c_t` only** (`L = L_flow + 0.1·L_var`) — **SIGReg / VICReg /
> covariance removed**; **(5)** context is **8 frames @256** (tubelet-2 → `N_ctx=1024`); **(6)**
> tubelet dropout on the encoder input is **removed**; **(7)** required monitors now include the
> **cross-video cosine similarity of `c_t`**; **(8)** multi-horizon prediction is **Phase 4**
> (deferred — see [`../PHASES/PHASE_4.md`](../PHASES/PHASE_4.md)). The original (pre-update) baseline
> is preserved verbatim in [`BRIEF_V0_1.md`](BRIEF_V0_1.md). Where this banner and §2.6 conflict with
> older prose elsewhere in this file, **the banner and §2.6 win.**
>
> **Purpose.** This is an agent-facing comprehension reference for the architecture defined in
> [`BRIEF_V0_2.md`](BRIEF_V0_2.md) (and historically the original PDF, preserved as
> [`BRIEF_V0_1.md`](BRIEF_V0_1.md)). It expands every component of the brief to the level of detail an implementation agent needs to write correct code: explicit shape contracts, exact gradient flow, exact training-stage semantics, exact constants. **There are no unresolved questions in this document.** Every choice not explicitly fixed by the brief has been resolved — either by external research with citation, or by deliberate decision recorded in §14. Where alternatives existed, one was chosen and the others discarded.
>
> **How to use.**
> - Read top to bottom once at the start of any implementation session.
> - §2 (symbol/shape table), §2.6 (locked constants), and §6 (stop-gradient table) are the most-referenced sections — re-consult whenever writing or modifying a function that touches latents, gradients, or sizes.
> - Do not deviate from §6, §7, §10 without explicit human approval. Those sections encode non-negotiable design constraints.
> - Cross-reference: every numerical constant in this document also appears in §2.6, which is the single source of truth. If two sections disagree, §2.6 wins and the disagreement is a bug to be fixed.

---

## 0. Document map

| § | Section | What the agent uses it for |
|---|---|---|
| 1 | Thesis in one paragraph | What we are building and what makes it different from a video diffusion model |
| 2 | Symbol and shape table | Every named tensor with its exact shape — single source of truth for shape contracts |
| 2.6 | Locked constants table | Every numerical constant in one place — single source of truth for values |
| 3 | The architecture in full | Component-by-component description of every learnable module |
| 4 | End-to-end forward-pass walkthrough | Shapes traced through the full pipeline for one batch |
| 5 | Losses | Every loss with full math, what it backpropagates into, and what it does not |
| 6 | Stop-gradient table | Complete enumeration of every stop-gradient in the system |
| 7 | EMA: what, where, how, schedule | How the target branch is maintained |
| 8 | Training schedule | What is trained, frozen, and gated at each of Stages 0–5, with concrete step counts |
| 9 | Runtime tests | Every diagnostic test, what it measures, what failure mode it catches, with concrete thresholds |
| 10 | Non-negotiable design constraints | The brief's §11, expanded with consequences-of-violation and code-level invariants |
| 11 | Design rationale | Why the architecture is shaped this way; what each piece exists to prevent or enable |
| 12 | Literature grounding | Each piece of the architecture traced back to its source paper |
| 13 | Dataset notes (SSv2) | Properties of Something-Something v2 relevant to implementation |
| 14 | Resolved decisions log | Every decision made beyond the brief, with the reasoning |

---

## 1. The thesis in one paragraph

We are training a **video world model** whose internal representation is a **hierarchy of two latents**: an abstract latent `c_t` carrying the future-relevant structure of a short context clip, and a detailed latent `e_t` carrying texture and local visual information. **`e_t` is produced by a frozen pretrained video encoder `E` (V-JEPA 2 ViT-L/16); `c_t` is produced by a trainable bottleneck `B` on top of it.** The model learns by predicting **future latents**, not future pixels: from `c_t` it predicts `c_plus` (the future abstract state), and from `e_t` plus a coarse condition it predicts `e_plus` (the future detailed state). Both predictions are performed by **flow-matching networks** that learn a continuous-time velocity field transforming Gaussian noise into the future target latent. The future abstract target is **clip-level**, `c_plus = c⁺_{t+k} = B_EMA(E(x_{≤t+k}))`: the **same frozen encoder** on the future clip, then a **slow-moving EMA copy of the bottleneck**. Targets are always stop-gradient. Because the encoder is frozen and shared, **only the bottleneck has an EMA copy** (`B_EMA`); there is no `E_bar`. **The frame decoder is a separate, later-stage module trained only after the latent world model is verified working; it never updates the encoder or bottleneck.** The single most important property of this design — the property all the bypass tests in §9 exist to verify — is that the abstract latent `c_t` must carry **real predictive signal** and must not collapse, copy, or be bypassed by the detailed latent or by the frame decoder.

This is **not** a video diffusion model. A normal video diffusion model trains a denoiser directly on pixel-space or VAE-latent-space targets; nothing forces it to develop a compressed predictive state. Here, the compressed state is the entire point, and the generative components (the flows, the frame decoder) exist only to teach and to render it.

---

## 2. Symbol and shape table

These are the canonical names and shapes used throughout the codebase. **Any function that produces or consumes any of these tensors must have a docstring shape contract matching this table exactly.** Batch dimension `B` is omitted from the "shape" column but is always present in code.

### 2.1 Inputs

| Symbol | What it is | Shape (without batch) | dtype | Origin |
|---|---|---|---|---|
| `X_ctx` | Context clip: the 8 frames immediately preceding the prediction target | `(T=8, C=3, H=256, W=256)` | float32 from loader, bf16 after AMP cast | Dataloader |
| `X_tgt` | Future clip ending at `t+k` (the prediction target clip) | `(T=8, C=3, H=256, W=256)` | float32 → bf16 | Dataloader |

**v0.2:** the prediction target is a **clip** `x_{≤t+k}` (length-`T` window ending at `t+k`), not a single frame. For Phases 1–3 a single fixed horizon `k` is used; Phase 4 generalizes to `k∈{4,8,16,32}`.

Pixel values use the **frozen encoder's expected normalization** (the V-JEPA processor / ImageNet stats), **not** [-1, 1]. The VAE's [-1, 1] range is a separate Stage-4 concern (see §13 and §14 #11/#27).

### 2.2 Latents produced by the encoder/bottleneck

| Symbol | What it is | Shape | dtype | Produced by | Branch |
|---|---|---|---|---|---|
| `e_t` | Detailed latent of the context clip | `(N_ctx=1024, D_e=1024)` | bf16 | **Frozen** encoder `E` | frozen (no grad) |
| `c_t` | Abstract latent of the context clip | `(N_c=32, D_c=256)` | bf16 | Online bottleneck `B` | online (trainable) |
| `e_plus` | Detailed latent of the future clip | `(N_tgt=1024, D_e=1024)` | bf16, **stop-grad** | **Same frozen** encoder `E` | frozen (no grad) |
| `c_plus` | Abstract latent of the future clip (`c⁺_{t+k}`) | `(N_c=32, D_c=256)` | bf16, **stop-grad** | EMA bottleneck `B_EMA` | target (no grad) |

Token counts derived from the encoder's patch geometry (V-JEPA 2: patch 16, **tubelet 2**):
- `N_ctx = (T/2) × (256 / 16)² = 4 × 256 = 1024` — 8 context frames merged into 4 temporal tokens × 16×16 spatial.
- `N_tgt = 1024` — the target is a clip with the **same geometry** as the context (clip-level target, v0.2).
- `D_e = 1024` — fixed by the frozen V-JEPA 2 ViT-L/16 embedding dimension.
- `N_c = 32` — fixed by the bottleneck's 32 learned query slots, regardless of input length.

**No tubelet dropout** in v0.2 (the frozen encoder never saw dropped tokens at pretraining), so `e_t` always has the full `N_ctx=1024` tokens. `e_t` is **frozen** (no grad); only `c_t` is trainable.

### 2.3 Predictions produced by the flow networks

| Symbol | What it is | Shape | dtype | Produced by | Notes |
|---|---|---|---|---|---|
| `z_c` | Noised version of `c_plus` at flow-time `τ_c` | `(N_c=32, D_c=256)` | bf16 | `(1-τ_c)·ε_c + τ_c·c_plus` | `ε_c ~ N(0, I)`, `τ_c ~ U(0,1)` |
| `u_c` | Ground-truth coarse velocity | `(N_c=32, D_c=256)` | bf16 | `c_plus - ε_c` | Constant along trajectory (rectified flow) |
| `u_c_hat` | Predicted coarse velocity | `(N_c=32, D_c=256)` | bf16 | `F_c(z_c, τ_c, c_t)` | Target of L_c |
| `c_hat` | One-step prediction of `c_plus`, used as coarse condition for F_e in Stage 3 | `(N_c=32, D_c=256)` | bf16, **stop-grad** when fed to F_e | `(z_c + (1 − τ_c) · u_c_hat).detach()` | One-step Euler — see §14 #5 |
| `z_e` | Noised version of `e_plus` at flow-time `τ_e` | `(N_tgt=64, D_e=1024)` | bf16 | `(1-τ_e)·ε_e + τ_e·e_plus` | `ε_e ~ N(0, I)`, `τ_e ~ U(0,1)`, independently sampled from `τ_c` |
| `u_e` | Ground-truth fine velocity | `(N_tgt=64, D_e=1024)` | bf16 | `e_plus - ε_e` |  |
| `u_e_hat` | Predicted fine velocity | `(N_tgt=64, D_e=1024)` | bf16 | `F_e(z_e, τ_e, e_t, c_cond)` | `c_cond` is `c_plus` (Stage 2) or `stopgrad(c_hat)` (Stage 3) |
| `e_hat` | Predicted future detailed latent; conditioning into D in Stage 4 | `(N_tgt=64, D_e=1024)` | bf16, **stop-grad** when fed to D | Full ODE rollout of F_e (4 Heun steps) at inference; one-step at Stage 4 training-time | See §14 #5 |

### 2.4 Frame-generator-stage quantities (Stage 4 only)

| Symbol | What it is | Shape | Notes |
|---|---|---|---|
| `a_y` | VAE-encoded future frame, patched for D | `(N_vae=64, D_vae_token=16)` | Underlying VAE latent is `(4, 16, 16)`; 2×2 spatial patching gives 8×8=64 tokens with 4×2×2=16 dims each |
| `z_x` | Noised patched VAE latent at flow-time `τ_x` | `(N_vae=64, D_vae_token=16)` | `(1-τ_x)·ε_x + τ_x·a_y` |
| `u_x` | Ground-truth velocity in patched VAE latent space | `(N_vae=64, D_vae_token=16)` | `a_y - ε_x` |
| `u_x_hat` | Predicted velocity | `(N_vae=64, D_vae_token=16)` | `D(z_x, τ_x, stopgrad(e_hat))`, after D's own unpatchify-projection |

The VAE used is `stabilityai/sd-vae-ft-mse` (frozen). **v0.2 (256×256):** it produces a `4×32×32`
latent; patchified with 2×2 spatial patches → **256 tokens** of dim 16, projected to 512 for D's
transformer blocks, then back to dim 16 and unpatchified. D's internal model dim is 512.

> The `N_vae=64` figures in the §2.4 table rows above are the **128-baseline** numbers; at 256 they
> become **256** tokens. These (and the exact `e_hat` token count D conditions on) lock when Phase 3
> is finalized against the §14 #35 detailed-target-geometry decision.

### 2.5 Flow-time scalars

| Symbol | What it is | Shape | Range | Sampling |
|---|---|---|---|---|
| `τ_c` | Coarse-flow time | `(B,)` | `[0, 1]` | Uniform U(0,1), sampled per example per step |
| `τ_e` | Fine-flow time | `(B,)` | `[0, 1]` | Uniform U(0,1), sampled per example per step, **independently** from `τ_c` |
| `τ_x` | Frame-flow time (Stage 4) | `(B,)` | `[0, 1]` | Uniform U(0,1) |

### 2.6 Locked constants table — single source of truth for numerical values

**Any time the agent needs a constant, look here.** If a value appears elsewhere in this document and conflicts with this table, that elsewhere is a bug.

#### Architecture

| Constant | Value | Where it's used |
|---|---|---|
| Number of context frames `T` | **8** | Context clip |
| Frame spatial size `H × W` | **256 × 256** | All frames |
| Encoder patch / tubelet (T × H × W) | **2 × 16 × 16** (V-JEPA 2 native) | Frozen encoder tokenization |
| `N_ctx` (context tokens) | **1024** | Frozen-encoder output `((8/2)·(256/16)²)` |
| `N_tgt` (target tokens) | **1024** | Future-**clip** target, same geometry |
| `N_c` (abstract tokens) | 32 | Bottleneck query slot count |
| `D_e` (detailed latent dim) | **1024** | Frozen encoder output, F_e memory |
| `D_c` (abstract latent dim) | 256 | Bottleneck output, F_c |
| Encoder | **Frozen V-JEPA 2 ViT-L/16** (depth 24, dim 1024, 16 heads, MLP 4, 3D-RoPE) | Shared by both branches; `requires_grad=False` |
| Encoder HF source | `facebook/vjepa2-vitl-fpc64-256` (V-JEPA 2 ViT-L) | Load + freeze; verify exact repo ID at load |
| Encoder native resolution | **256 (matches our resolution exactly)** | Frozen encoder input |
| Bottleneck input projection | 1024 → 256 | First proj in B (`D_e → d_c`) |
| Bottleneck ConvNeXt blocks | 2 | Before cross-attention in B |
| Bottleneck cross-attn heads | 8 | In B |
| F_c blocks | 6 | Coarse flow |
| F_c dim | 256 | Coarse flow internal |
| F_c heads | 8 | Coarse flow |
| F_e blocks | 8 | Fine flow |
| F_e dim | 384 | Fine flow internal |
| F_e heads | 8 | Fine flow |
| D blocks | 12 | Frame generator |
| D dim | 512 | Frame generator internal |
| D heads | 8 | Frame generator |
| D MLP ratio | 4 | Frame generator |
| Frame VAE | `stabilityai/sd-vae-ft-mse` | Frozen image VAE for Stage 4 |
| VAE downsampling | 8× | Yields 32×32 latent for 256×256 input (v0.2) |
| VAE latent channels | 4 | Single source of truth |

#### Training

| Constant | Value | Source |
|---|---|---|
| Total training steps | 150,000 | Brief §8 |
| Stage 1 steps | 30,000 | §14 #7 |
| Stage 2 steps | 25,000 | §14 #7 |
| Stage 3 steps | 50,000 | §14 #7 |
| Stage 4 steps | 45,000 | §14 #7 |
| Stage 3 `c_cond` ramp (c_plus → c_hat), linear | 5,000 steps | §14 #8 |
| Global batch size | 64 clips | Brief §7 |
| Optimizer | AdamW | Brief §7 |
| AdamW betas | (0.9, 0.95) | Brief §7 |
| Weight decay | 0.05 | Brief §7 |
| Gradient clipping (global norm) | 1.0 | Brief §7 |
| Precision | bf16 AMP | Brief §7 |
| LR (encoder E) | **n/a — frozen** | v0.2 (§14 #28) |
| LR (bottleneck B) | 2e-4 | Brief §7 |
| LR (F_c) | 4e-4 | Brief §7 |
| LR (F_e) | 4e-4 | Brief §7 |
| LR (frame generator D) | 2e-4 | Brief §7 |
| LR schedule (latent stages 1–3 combined) | 10k warmup, cosine decay over remaining 95k | Brief §7 + §14 #7 |
| LR schedule (Stage 4) | 3k warmup, cosine decay over remaining 42k | §14 #7 |
| Tubelet dropout (context only) | **removed (0%)** | v0.2 (§14 #29) — frozen encoder |
| Condition dropout (flows) | 10% | Brief §7 |
| EMA momentum `m` start | 0.996 | Brief §3 |
| EMA momentum `m` end | 0.9999 | Brief §3 |
| EMA scope | **bottleneck only (`B → B_EMA`)** | v0.2 (§14 #28) — encoder frozen |
| EMA cosine schedule denominator `S` | 105,000 (sum of latent stages) | §14 #9 |

#### Losses

| Constant | Value | Source |
|---|---|---|
| `λ_fine` (L_e weight in total loss) | 1.0 | Brief §4.3 |
| `λ_var` (variance-floor weight) | **0.10** | v0.2 supervisor (§14 #30) |
| Variance-floor std target | **1.0** (hinge: `max(0, 1.0 − Std(c_j))`) | v0.2 supervisor |
| Variance floor applies to | **`c_t` only** | v0.2 supervisor |
| SIGReg / VICReg / covariance loss | **removed (none)** | v0.2 supervisor (§14 #30) |

#### Diagnostic thresholds (§9)

| Threshold | Value |
|---|---|
| F_c vs copy-baseline (val L_c ratio) after Stage 1 step 10k | ≤ 0.70 |
| F_c vs batch-mean baseline (val L_c ratio) after Stage 1 step 10k | ≤ 0.50 |
| Shuffled-c test (L_e ratio shuffled/real) by end Stage 2 | ≥ 1.5 |
| Shuffled-c test by end Stage 3 | ≥ 2.0 |
| Per-dim std collapse warning (`c_t`) | >15% of dims with std < 0.1× median |
| Per-dim std collapse hard stop (`c_t`) | >30% of dims |
| **Cross-video cosine of `c_t` — healthy** | **well below ~0.5 (distinct videos → distinct `c_t`)** |
| **Cross-video cosine of `c_t` — concerning** | **drifting toward 1.0** |
| Effective rank `c_t` (D=256) — healthy | > 60 |
| Effective rank `c_t` — concerning | < 20 |
| Effective rank `c_t` — hard stop | < 5 |
| Effective rank `e_t` (D=1024) — reference only | frozen encoder; `e_t` cannot collapse (not trained) |

#### Data

| Constant | Value | Source |
|---|---|---|
| Frame stride (within a clip) | 2 | §14 #12 |
| Pixel normalization | **encoder's expected normalization** (V-JEPA processor); VAE uses [-1,1] separately in Stage 4 | §14 #27 |
| Resize policy (train) | shorter-side to **256**, then random-crop to **256×256** | §14 R9 (v0.2) |
| Resize policy (eval) | shorter-side to **256**, then center-crop to **256×256** | §14 R9 (v0.2) |
| Horizontal flip | OFF | §14 R10 |
| Temporal flip | OFF | §14 R10 |
| Color jitter | brightness=0.4, contrast=0.4, saturation=0.4 (no hue) | §14 R10 |

---

## 3. The architecture in full

For each module: input, output, parameter budget, behavioral requirements. Every implementation choice is fixed in this section — there are no "either/or"s.

### 3.1 Tokenization — handled by the frozen encoder

In v0.2 we **do not implement our own patchifier or position embeddings.** Tokenization, the 3D
RoPE position encoding, and the spatio-temporal patch/tubelet projection are all **internal to the
frozen V-JEPA 2.1 encoder**. We feed it pixel clips (after its own preprocessing/normalization) and
read out per-tubelet features.

- **Context input:** `(B, T=8, C=3, 256, 256)` → encoder → `e_t` of shape `(B, N_ctx=1024, D_e=1024)`.
- **Target input:** `(B, T=8, C=3, 256, 256)` (future clip ending at `t+k`) → **same frozen encoder**
  → `e_plus` of shape `(B, N_tgt=1024, D_e=1024)`.
- Patch geometry (V-JEPA 2): patch 16, **tubelet 2** → `(8/2) × (256/16) × (256/16) = 4 × 16 × 16 =
  1024` tokens. 3D-RoPE is relative and tolerates our off-native resolution (256 vs native 384) and
  short clip length.
- The encoder's own input normalization (via its HF `AutoVideoProcessor`) is used — **not** [-1,1].

### 3.2 Encoder `E` — frozen pretrained V-JEPA 2 ViT-L/16

`E` is the **pretrained, frozen** video encoder. We do **not** train it and there is **no** separate
target encoder `E_bar` — the same frozen `E` is applied to both the context clip and the future clip.

- **Source:** V-JEPA 2 ViT-L/16 (HF `vjepa2` family; repo `facebook/vjepa2-vitl-fpc64-256`).
  Embedding dim **1024**, depth 24, 16 heads, MLP ratio 4, 3D-RoPE, no [CLS]. Native resolution
  **256** — exactly our input resolution, so no off-native RoPE stretch.
  > The originally-picked ViT-B/16 (`D_e=768`) has **no `transformers` repo** (torch.hub only); ViT-L
  > via HF is the verified clean path and its native 256 matches our resolution (see
  > [`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md) decision banner).
- **Load + freeze:** `requires_grad=False` on all params; always `eval()`; forward under
  `torch.no_grad()`.
- **Input:** context (or future) clip `(B, 8, 3, 256, 256)`.
- **Output:** `e_t` (or `e_plus`) of shape `(B, 1024, 1024)` — `last_hidden_state` /
  `get_vision_features(...)`.
- **Parameter budget:** ≈ 80M (frozen — not in any optimizer group).
- **API sketch:** see [`FROZEN_ENCODER_RESEARCH.md`](FROZEN_ENCODER_RESEARCH.md) §7.

#### 3.2.1 No tubelet dropout (v0.2)

Tubelet dropout is **removed.** A frozen pretrained encoder never saw dropped tokens at pretraining,
so masking its input feeds it out-of-distribution data. `e_t` therefore always has the full
`N_ctx=1024` tokens. If a shortcut-prevention knob is wanted later, apply light masking **inside the
trainable bottleneck**, not at the encoder input (decision: §14 #29).

### 3.3 Bottleneck `B`

Compresses `e_t` (1024 context tokens of dim **1024**) into `c_t` (32 abstract tokens of dim 256). The **same architecture** compresses `e_plus` (1024 future-clip tokens of dim 1024) into `c_plus` (32 tokens of dim 256) on the EMA branch. **This is the only trainable module on the encoder side.**

**Structure (fixed):**

0. **Input projection 1024 → (mixer width).** First project the frozen `e_t` from `D_e=1024` down to
   the mixer/working width before ConvNeXt. (No kept-mask, no zero-fill — tubelet dropout is removed
   in v0.2, so all 1024 tokens are real.)

1. **Spatial-grid reconstruction + per-temporal-slot 2D ConvNeXt mixing (2 blocks).**
   - Reshape the 1024 tokens into a `(B, 4, 16, 16, ·)` grid (4 temporal tokens × 16×16 spatial), then
     to `(B·4, 16, 16, ·)` so each of the 4 temporal slots is a separate "image" for the conv.
   - Apply 2 ConvNeXt-V2-style blocks: `Conv2d` 7×7 depthwise → LayerNorm → pointwise linear
     (×4 expansion) → GELU → pointwise linear, with a residual. Each block has its own parameters;
     weights shared across the 4 temporal-slot grids (and applied identically to the target grid).
   - Flatten spatial back to `(B, 1024, ·)`.

2. **Cross-attention to 32 learned query embeddings.** A single cross-attention layer:
   - Queries: 32 learnable embeddings of dim 256 (parameters of B), broadcast to batch.
   - Keys, values: post-mixer tokens projected to dim 256, used as `(N_input=1024, 256)` keys/values.
   - 8 attention heads.
   - Output: 32 tokens of dim 256.

3. **Output MLP block.** One transformer-style block on the 32 output tokens: LayerNorm → linear
   256→1024 → GELU → linear 1024→256, with residual. Final LayerNorm.

**Parameter budget:** ≈ 4–5M (slightly larger than v0.1 due to the 1024→256 input projection).

**Outputs:**
- Online branch: `c_t = B(e_t)` — `(32, 256)`.
- Target branch: `c_plus = B_EMA(e_plus)` — `(32, 256)`, then `.detach()`.

### 3.4 EMA target branch (`B_EMA` only)

In v0.2 the encoder is **frozen and shared**, so there is **no `E_bar`** — the same frozen `E`
produces both `e_t` (context) and `e_plus` (future clip). Only the **bottleneck** has an EMA copy.

- `E` (shared, frozen): future clip `(B, 8, 3, 256, 256)` → `e_plus` `(B, 1024, 1024)`.
- `B_EMA`: takes `e_plus`. Outputs `c_plus` `(B, 32, 256)`, then `.detach()` (stop-grad).

**Implementation requirements:**
- `B_EMA` is a separate `nn.Module` copy of `B`; independent parameters kept aligned by the EMA rule.
- `param.requires_grad = False` on every `B_EMA` parameter; permanently `eval()`.
- Initialized at Stage 0 by copying `B.state_dict()` into `B_EMA`. (No encoder copy needed.)
- EMA updates run on `B → B_EMA` only (§7).

### 3.5 Symmetry between context and target representations (v0.2)

Both branches now use the **same frozen encoder** on length-`T` clips, so context and target have
**identical geometry** (`1024` tokens, dim `1024`). The target is the future clip `x_{≤t+k}` (the
window ending `k` frames later). The bottleneck (`B` online, `B_EMA` target) is the only difference
between the two paths. This is cleaner than v0.1's context/target asymmetry and follows the
supervisor's `c⁺_{t+k} = B_EMA(E(x_{≤t+k}))` target definition. The I-JEPA principle still holds —
the target representation is produced by a stable (EMA) module and is stop-gradient.

### 3.6 Coarse flow `F_c`

Learn the velocity field of a continuous-time flow from Gaussian noise to `c_plus`, conditioned on `c_t`.

**Structure (fixed):**

- 6 transformer blocks, dim 256, 8 heads, MLP ratio 4. ≈ 5M parameters.
- Each block is DiT-style: self-attention + MLP, both gated by adaLN-Zero modulation from the `τ_c` time embedding.
- **Time conditioning (adaLN-Zero):**
  1. Embed `τ_c` via sinusoidal Fourier features → 256-dim vector.
  2. Project through a 2-layer MLP (256 → 1024 → 256) to a time vector.
  3. Per block, project the time vector to 6 modulation parameters: pre-attn scale, post-attn scale, post-attn shift, pre-MLP scale, post-MLP scale, post-MLP shift.
  4. **Zero-initialize** the final projection layer of step 3 so all modulation parameters are zero at training start — the block is then identity at step 0 (DiT's "adaLN-Zero" initialization).
- **Conditioning on `c_t`:** concatenate `c_t` (32 tokens, dim 256) with `z_c` (32 tokens, dim 256) into a single 64-token sequence at the input of the first block. Self-attention within and across both halves provides the conditioning pathway. At the output of the final block, read out **only the first 32 tokens** (the `z_c` half) and treat them as `u_c_hat`.
- **Condition dropout:** with probability `p_cond = 0.10` per training example, replace `c_t` with a learned null embedding of shape `(32, 256)` (one learned parameter of F_c) before concatenation. See §14 #10.

**Output:** `u_c_hat` of shape `(32, 256)` — predicted velocity.

### 3.7 Fine flow `F_e`

Learn the velocity field of a continuous-time flow from Gaussian noise to `e_plus`, conditioned on `e_t` and a coarse condition `c_cond`.

**Structure (fixed):**

- 8 transformer blocks, dim 384, 8 heads, MLP ratio 4. ≈ 14M parameters.
- Each block is DiT-style with **self-attention → cross-attention → MLP**, all gated by adaLN-Zero modulation from the `τ_e` time embedding (same recipe as F_c).
- **Self-attention** runs over the 64 noised-target tokens (`z_e`).
- **Cross-attention** is the conditioning pathway:
  - Build a combined memory: project `c_cond` from `(32, 256)` to `(32, 384)` via a single linear layer to get `proj_c_cond`. Concatenate with `e_t` to form `(N_ctx_post + 32, 384)`.
  - Cross-attention from `z_e` queries (64 tokens) to this memory (keys/values).
- **Condition dropout:** with probability `p_cond = 0.10` per training example, replace **the entire memory** with a learned null memory (a learned parameter of shape `(max_memory_len, 384)` truncated to the actual memory length) before cross-attention. The drop is a single joint Bernoulli for both `e_t` and `proj_c_cond` together — not independent.

**Inputs in code:**
- `z_e` of shape `(64, 384)` — the noised target latent at flow-time `τ_e`.
- `τ_e` — scalar per example.
- `e_t` of shape `(N_ctx_post, 384)` — context conditioning.
- `c_cond` of shape `(32, 256)` — coarse condition. `c_plus` in Stage 2 (teacher-forced); `stopgrad(c_hat)` in Stage 3 (after the ramp); a linear mix during the Stage 3 ramp (see §8).

**Output:** `u_e_hat` of shape `(64, 384)` — predicted velocity.

### 3.8 Frame generator `D`

Render the predicted future frame from `stopgrad(e_hat)`, in the latent space of the frozen `sd-vae-ft-mse` VAE.

**Structure (fixed):**

- 12 transformer blocks, dim 512, 8 heads, MLP ratio 4. ≈ 38M parameters.
- Each block is DiT-style: **self-attention → cross-attention → MLP**, with adaLN-Zero on `τ_x`.
- **Input pipeline:**
  1. VAE latent `a_y` of shape `(4, 16, 16)` is patched with **2×2 spatial patches** → 8×8 = 64 tokens of dim `4 × 2 × 2 = 16`.
  2. A linear layer projects these 16-dim tokens to D's internal model dim 512.
  3. Add 2D sin-cos position embeddings over the 8×8 token grid.
- **Self-attention** over the 64 VAE-latent tokens.
- **Cross-attention** from these 64 queries against `stopgrad(e_hat)` (64 tokens, dim 384), projected to dim 512 via a linear layer.
- **Output pipeline:** linear projection from dim 512 back to dim 16, then unpatchify to `(4, 16, 16)`.

**Loss:** flow matching against `a_y = VAE.encode(y).latent_dist.mode()` — see §5.5.

**Trained alone in Stage 4** — encoder, bottleneck, flows, VAE are all frozen.

---

## 4. End-to-end forward-pass walkthrough

This section traces one training batch through the **Stage 3** latent path — the most complete latent stage (coarse + fine flows, predicted-coarse conditioning after ramp). Shapes omit batch dim `B` unless noted. Every `.detach()` / `as_target()` placement is explicit.

**Batch from dataloader:**
- `X_ctx`: `(8, 3, 256, 256)` — context clip, encoder-normalized
- `X_tgt`: `(8, 3, 256, 256)` — future clip ending at `t+k`, encoder-normalized

### Step 1 — Frozen encoder on context (no grad)

```
with torch.no_grad():
    e_t = E(X_ctx)        # frozen V-JEPA 2 ViT-L/16; e_t: (1024, 1024)
```

No patchifier / pos-embed / tubelet dropout on our side — all internal to the frozen encoder.

### Step 2 — (folded into Step 1; `e_t` is the frozen encoder output)

### Step 3 — Online bottleneck (trainable)

```
c_t = B(e_t)
# c_t: (32, 256)
```

### Step 4 — Target branch (no grad)

```
with torch.no_grad():
    e_plus = E(X_tgt)                  # SAME frozen encoder; (1024, 1024)
    c_plus = as_target(B_EMA(e_plus))  # (32, 256), detached  (= c⁺_{t+k})
    e_plus = as_target(e_plus)         # (1024, 1024), detached
```

### Step 5 — Sample flow noise and times

```
eps_c ~ N(0, I)   # (32, 256)
eps_e ~ N(0, I)   # (64, 384)
tau_c ~ U(0, 1)   # scalar per example, shape (B,) in code
tau_e ~ U(0, 1)   # independent from tau_c
```

### Step 6 — Coarse flow interpolation

```
z_c = (1 - tau_c) * eps_c + tau_c * c_plus    # (32, 256); c_plus is stop-grad
u_c = c_plus - eps_c                            # (32, 256); ground-truth velocity
```

Broadcast `tau_c` to `(B, 1, 1)` for the interpolation.

### Step 7 — Coarse flow prediction

```
u_c_hat = F_c(z_c, tau_c, c_t)   # (32, 256)
# Inside F_c: optional 10% condition dropout replaces c_t with learned null
```

### Step 8 — One-step c_hat for F_e conditioning (Stage 3)

```
c_hat = z_c + (1 - tau_c) * u_c_hat   # (32, 256)
# Do NOT detach yet — L_c backprop must reach F_c through u_c_hat
```

### Step 9 — Stage 3 coarse condition (after ramp)

```
alpha = min(1.0, (step - 55000) / 5000)   # linear ramp over first 5k of Stage 3
c_cond = (1 - alpha) * c_plus + alpha * as_target(c_hat)   # (32, 256)
# At alpha = 1: c_cond = as_target(c_hat) only
```

Stage 2 uses `c_cond = c_plus` throughout (teacher-forced).

### Step 10 — Fine flow interpolation

```
z_e = (1 - tau_e) * eps_e + tau_e * e_plus   # (64, 384); e_plus stop-grad
u_e = e_plus - eps_e                            # (64, 384)
```

### Step 11 — Fine flow prediction

```
proj_c = Linear(256→384)(c_cond)              # (32, 384)
memory = concat(e_t, proj_c, dim=0)             # (N_ctx_post + 32, 384)
u_e_hat = F_e(z_e, tau_e, memory)               # (64, 384)
# F_e: self-attn on z_e, cross-attn to memory; 10% joint condition dropout
```

### Step 12 — Losses (Stage 3)

```
L_c = mean((u_c_hat - u_c) ** 2)
L_e = mean((u_e_hat - u_e) ** 2)
L_var = variance_floor(c_t)        # (1/d) Σ_j max(0, 1.0 - Std(c_j)); c_t only
L = L_c + 1.0 * L_e + 0.10 * L_var
```

See §5 for gradient routing. (No SIGReg — removed in v0.2.)

### Step 13 — Backward + clip + optimizer step

```
L.backward()
clip_grad_norm_(all_trainable_params, 1.0)
optimizer.step()
optimizer.zero_grad()
```

Trainable in Stage 3: `B`, `F_c`, `F_e`. Not trainable: **frozen `E`**, `B_EMA`, detached targets.

**Verify:** `c_hat` path into `F_e` uses `as_target(c_hat)` so **no gradient from L_e reaches F_c**.

### Step 14 — EMA update (latent stages only; bottleneck only)

```
m = ema_cosine(step, start=0.996, end=0.9999, total=105_000)
for p_online, p_ema in zip(B.parameters(), B_EMA.parameters()):
    p_ema.lerp_(p_online, 1 - m)
# No encoder EMA — E is frozen and shared.
```

No EMA update during Stage 4 (bottleneck frozen too).

### Stage 4 forward (frame generator) — abbreviated

With `E`, `B`, `F_c`, `F_e`, VAE frozen:

```
with torch.no_grad():
    e_t, c_t = online_path(X_ctx)
    c_hat = heun_rollout(F_c, c_t, steps=4)      # inference integrator
    e_hat = heun_rollout(F_e, e_t, c_hat, steps=4)
    e_hat = as_target(e_hat)

a_y = patchify(vae.encode(y).mode())             # (64, 16), frozen VAE
eps_x ~ N(0,I); tau_x ~ U(0,1)
z_x = (1 - tau_x) * eps_x + tau_x * a_y
u_x = a_y - eps_x
u_x_hat = D(z_x, tau_x, e_hat)                   # only D has grad
L_frame = mean((u_x_hat - u_x) ** 2)
```

Training-time Stage 4 may use one-step `e_hat` for speed (see `AGENT_FILES/PHASES/PHASE_3.md` §6.3); inference always uses Heun (4 steps).

---

## 5. Losses

All flow-matching losses use **rectified flow** (linear interpolation CFM): `z = (1-τ)·ε + τ·x_target`, `u = x_target - ε`, minimize `||F(z, τ, cond) - u||²`. Reference: Lipman et al., Flow Matching for Generative Modeling (arXiv:2210.02747).

### 5.1 Coarse JEPA-flow loss `L_c`

```
ε_c ~ N(0, I)
τ_c ~ U(0, 1)
z_c = (1 - τ_c) · ε_c + τ_c · c_plus
u_c = c_plus - ε_c
L_c = mean_square(F_c(z_c, τ_c, c_t) - u_c)
```

| Receives gradients from L_c | Does NOT receive gradients |
|---|---|
| `B`, `F_c` | `E` (frozen), `B_EMA`, `c_plus`, `e_plus` |

`c_t` is **not** stop-gradient on the conditioning path — the **bottleneck** must learn predictive abstract states (the encoder `E` is frozen, so the learning happens in `B`).

### 5.2 Fine JEPA-flow loss `L_e`

**Stage 2 (teacher-forced):** `c_cond = c_plus` (already detached from EMA).

**Stage 3 (predicted-coarse):** `c_cond = as_target(c_hat)` where `c_hat = z_c + (1-τ_c)·u_c_hat`.

```
ε_e ~ N(0, I)
τ_e ~ U(0, 1)          # sampled independently from τ_c
z_e = (1 - τ_e) · ε_e + τ_e · e_plus
u_e = e_plus - ε_e
L_e = mean_square(F_e(z_e, τ_e, e_t, c_cond) - u_e)
```

| Receives gradients from L_e | Does NOT receive gradients |
|---|---|
| `E`, `B`, `F_e` | `c_plus`, `e_plus`, `c_hat` (when used as c_cond), `F_c` |

`F_c` is excluded because `c_hat` is detached before entering `F_e`. **Violating this turns the abstract latent into a texture carrier** (brief §11).

During Stage 3 ramp, `c_cond` is a convex mix; only the `c_hat` portion is detached — the `c_plus` portion is already stop-grad from EMA.

### 5.3 Variance floor — collapse prevention (v0.2; replaces SIGReg)

**SIGReg / VICReg / covariance losses are removed.** Collapse prevention is minimal: a **variance
floor on `c_t` only** (supervisor directive). The frozen encoder cannot collapse, so `e_t` needs no
regularization; only the bottleneck output `c_t` can collapse.

Mechanism:
1. Flatten `c_t` across slots/features per batch (so each of the `d` feature dimensions has a column
   of values across the batch).
2. Compute the **per-dimension standard deviation** `Std(c_j)`.
3. Hinge each dimension at a target std of **1.0**: `max(0, 1.0 − Std(c_j))`.
4. Average over the `d` dimensions.

```
L_var = (1 / d) · Σ_j max(0, 1.0 − Std(c_j))
L_reg = λ_var · L_var,   λ_var = 0.10   (i.e. total adds 0.1 · L_var)
```

`L_var` is active from **Stage 1 step 0** (on `c_t`). It is a **guardrail, not a teacher**: it only
prevents `c_t` from going constant. The **flow-matching objective** is what makes `c_t` meaningful.
Do **not** over-weight `λ_var` or add covariance/SIGReg terms (§14 #30).

### 5.4 Total latent objective (Stages 1–3)

```
L_latent = L_c + λ_fine · L_e + 0.1 · L_var(c_t)
λ_fine = 1.0
```

(`L_c` is the coarse flow-matching loss `L_flow` of §5.1, conditioned on `c_t`.)

**Stage 1 only:** `L_latent = L_c + 0.1 · L_var(c_t)` (no L_e). This matches the supervisor's
`L = L_flow + 0.1·L_var` for the coarse stage.

### 5.5 Frame generator loss `L_frame` (Stage 4 only)

```
a_y = patchify(VAE.encode(y).latent_dist.mode())   # frozen VAE
ε_x ~ N(0, I);  τ_x ~ U(0, 1)
z_x = (1 - τ_x) · ε_x + τ_x · a_y
u_x = a_y - ε_x
L_frame = mean_square(D(z_x, τ_x, as_target(e_hat)) - u_x)
```

| Receives gradients | Does NOT receive gradients |
|---|---|
| `D` only | `E`, `B`, `F_c`, `F_e`, VAE, `e_hat` |

---

## 6. Stop-gradient table

Complete enumeration. Use centralized `as_target(x)` helper in code (`AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §6).

| # | Tensor / module | Stop-grad? | Where | Rationale |
|---|---|---|---|---|
| 0 | Encoder `E` | **Frozen — no grad ever** | All stages, both branches | Pretrained, never updated |
| 1 | `e_plus` (= `e_{t+k}`) | **Always yes** | After frozen `E` on future clip | Target side — predicted, never optimized |
| 2 | `c_plus` (= `c⁺_{t+k}`) | **Always yes** | After `B_EMA` output | EMA target — never optimized |
| 3 | `c_hat` → `F_e` input | **Yes** | Stage 3+ before `c_cond` | Prevents L_e from backprop into F_c |
| 4 | `e_hat` → `D` input | **Yes** | Stage 4 always | Frame gen must not rewrite world model |
| 5 | `B_EMA` | **No backprop ever** | All stages | EMA update only |
| 6 | `c_t` on flow conditioning | **No** | Latent training | Bottleneck must learn predictive structure |
| 7 | `u_c_hat` → `c_hat` → `F_e` | **Partial** | `c_hat` detached; `u_c_hat` not detached w.r.t. L_c | L_c trains F_c; L_e must not |
| 8 | VAE encode/decode | **Always yes** | Stage 4 | Frozen pretrained VAE |
| 9 | Latent stack in Stage 4 | **Frozen (no grad)** | `requires_grad=False` | Only D trains |

**Invariant:** The encoder `E` and `B_EMA` are detached/frozen on every path. `e_t` is from a frozen
encoder (already no-grad). `c_t` from the trainable `B` is **not** detached on the conditioning path
(except `c_hat` into `F_e`, row 3).

---

## 7. EMA: what, where, how, schedule

### What is EMA'd (v0.2)

Only the **bottleneck**: `B_EMA` (target bottleneck). The encoder is **frozen** (nothing to EMA — it
is identical on both branches). Flow networks and frame generator are **not** EMA'd.

### Initialization (Stage 0)

At training start, before any optimizer step:

```
B_EMA.load_state_dict(B.state_dict())
# Encoder E is loaded pretrained and frozen; no target-encoder copy exists.
```

### Update rule (after each optimizer step, Stages 1–3)

```
m = ema_cosine(step, m_start=0.996, m_end=0.9999, S=105_000)
θ_B_EMA ← m · θ_B_EMA + (1 - m) · θ_B
```

Implemented as `p_ema.lerp_(p_online, 1 - m)` in PyTorch, over `B`'s parameters only.

The schedule is **parametric** (cosine in step index). `S = 105_000` = sum of latent training stages
(30k + 25k + 50k). During Stage 4, EMA updates **do not run** (bottleneck frozen too).

### Why EMA targets exist

The bottleneck chases a moving target if trained against its own immediate outputs. The slow EMA copy
provides **stable future abstract representations** to predict, following I-JEPA / V-JEPA practice.
Without EMA + stop-grad, representations collapse or chase the predictor every step. (In v0.2 only the
bottleneck can move, since the encoder is frozen — so only the bottleneck needs the EMA safeguard.)

---

## 8. Training schedule

Stages 0–5 from the brief. **v0 implements Stages 0–4 only.** Stage 5 is optional polish — deferred post-v0.

### Cumulative step boundaries

| Stage | Global step range | Step count | Implementation phase |
|---|---|---|---|
| 0 | (sanity script, 0 train steps) | — | Phase 1 |
| 1 | `[0, 30_000)` | 30,000 | Phase 1 |
| 2 | `[30_000, 55_000)` | 25,000 | Phase 2 |
| 3 | `[55_000, 105_000)` | 50,000 | Phase 2 |
| 4 | `[105_000, 150_000)` | 45,000 | Phase 3 |
| 5 | post-v0 | — | Out of scope |

### Stage 0 — Setup

**Train:** nothing. **Loss:** none.

Pass condition: dataloader loads batches; all modules construct; **frozen `E` loads with `requires_grad=False`**; `B_EMA` initialized from `B`; one synthetic forward + backward + EMA update completes without NaN; bf16 AMP stable.

### Stage 1 — Coarse dynamics

**Train:** `B`, `F_c`. **Frozen:** `E` (pretrained), `B_EMA` (no grad, EMA updated).

**Loss:** `L_c + 0.1·L_var(c_t)`  (= supervisor's `L_flow + 0.1·L_var`).

**Pass condition:**
- `c_t` non-collapsed (effective rank > 60, variance health, cross-video cosine well below ~0.5).
- `F_c` beats **copy baseline** (val L_c ratio ≤ 0.70) and **batch-mean baseline** (≤ 0.50) after step ≥ 10k.

Copy baseline = predict future abstract state equals current `c_t`. Batch-mean = predict `c_plus` as mean of batch's `c_plus`.

### Stage 2 — Fine teacher-forcing

**Train:** `B`, `F_c`, `F_e` (E frozen). **Condition:** `c_cond = c_plus`.

**Loss:** full `L_latent`.

**Pass condition:** `F_e` works with true stopped `c_plus`; shuffled-c ratio ≥ **1.5** by step 55k.

**Ablation:** zeroing `c_plus` in `F_e` must hurt `L_e` — if not, hierarchy is decorative.

### Stage 3 — Predicted-coarse fine training

**Train:** `B`, `F_c`, `F_e` (E frozen). **Condition ramp:** linear mix from `c_plus` to `as_target(c_hat)` over first **5,000** steps of Stage 3 (steps 55k→60k), then 100% predicted-coarse.

**Loss:** full `L_latent`.

**Pass condition:** shuffled-c ratio ≥ **2.0** by step 105k; teacher-forced vs predicted-coarse gap narrows but predicted path remains viable.

### Stage 4 — Frame generator

**Train:** `D` only. **Frozen:** `E`, `B`, `F_c`, `F_e`, VAE.

**Loss:** `L_frame` only.

**LR:** 2e-4 with **3k warmup**, cosine over remaining 42k (separate from latent schedule).

**Pass condition:** decoder dependency test — shuffled `e_hat` degrades generation; frames visually depend on world-model latents.

### Stage 5 — Optional polish (not v0)

Brief: small LR fine-tune of `D`, maybe brief `F_e` unfreeze. Enter only if all ablations pass. **Not implemented in v0.**

### Optimizer summary (latent stages)

| Module | LR | AdamW β | Weight decay |
|---|---|---|---|
| `E` | **frozen (no optimizer group)** | — | — |
| `B` | 2e-4 | (0.9, 0.95) | 0.05 |
| `F_c` | 4e-4 | (0.9, 0.95) | 0.05 |
| `F_e` | 4e-4 | (0.9, 0.95) | 0.05 |
| `D` | 2e-4 | (0.9, 0.95) | 0.05 |

Latent LR schedule: **10k warmup**, cosine decay over remaining **95k** steps (Stages 1–3 combined). Global batch **64** clips; gradient accumulation allowed if OOM.

---

## 9. Runtime tests and early-stop gates

Run every **500–1,000** steps on a **fixed validation batch** (cache at init). Thresholds from §2.6.

> **v0.2 required minimum monitors (supervisor):** variance of `c_t` (§9.1), cross-video cosine
> similarity of `c_t` (§9.1b), effective rank of `c_t` (§9.2). All on `c_t` (the only thing that can
> collapse — the encoder is frozen).

### 9.1 Variance of `c_t`

**Measure:** per-dimension std of `c_t` across the val batch (the same quantity the variance floor
acts on, §5.3); report mean and the fraction of near-zero dims.

**Failure:** many dimensions → 0 or flatline.

| Level | Condition |
|---|---|
| Warning | > 15% of dims with std < 0.1× median std |
| Hard stop | > 30% of dims |

### 9.1b Cross-video cosine similarity of `c_t`

**Measure:** flatten `c_t` per video; compute mean pairwise cosine similarity across **different**
videos in the val batch.

**Failure:** value approaches 1.0 — all videos map to nearly the same direction (collapse that
variance alone misses).

| Level | Condition |
|---|---|
| Healthy | mean pairwise cosine well below ~0.5 |
| Concerning | drifting toward 1.0 |

### 9.2 Effective rank of `c_t`

**Measure:** covariance effective rank of flattened `c_t`.

| Latent | Healthy | Concerning | Hard stop |
|---|---|---|---|
| `c_t` (D=256) | > 60 | < 20 | < 5 |

(`e_t` is from a frozen encoder and cannot collapse; its rank is a reference quantity only.)

### 9.3 Coarse baseline (F_c vs trivial predictors)

**Measure:** val `L_c` for model vs copy vs batch-mean baselines.

| Test | Pass (after Stage 1 step ≥ 10k) |
|---|---|
| F_c vs copy | model L_c ≤ **0.70** × copy L_c |
| F_c vs batch-mean | model L_c ≤ **0.50** × batch-mean L_c |

### 9.4 Shuffled-c test — **central contract**

**Measure:** `L_e` with real `c_hat` (or `c_plus` in Stage 2) vs `L_e` with `c_hat` permuted across batch.

**Failure:** shuffled performs almost as well as real → `F_e` ignores abstract conditioning → **entire architecture invalid**.

| Checkpoint | Required ratio (shuffled / real) |
|---|---|
| End Stage 2 (step 55k) | ≥ **1.5** |
| End Stage 3 (step 105k) | ≥ **2.0** |

### 9.5 Teacher vs predicted gap

**Measure:** `L_e` with `c_plus` vs `L_e` with `as_target(c_hat)` on same batch.

**Failure:** predicted-coarse never approaches teacher-forced — ramp or F_c quality problem.

### 9.6 Gradient health

**Measure:** global grad norm, NaN in loss/grads, loss spikes.

**Failure:** repeated NaNs, exploding norms, EMA drift without recovery. Log every step; alert on threshold.

### 9.7 Decoder dependency (Stage 4+)

**Measure:** `L_frame` or visual quality with true vs shuffled `e_hat`.

**Failure:** shuffled `e_hat` still yields plausible frames → `D` bypasses world model → degenerate image generator.

---

## 10. Non-negotiable design constraints

From brief §11, expanded with **consequences of violation** and **code invariants**.

| Constraint | Consequence if violated | Code invariant |
|---|---|---|
| Encoder pretrained + frozen | System becomes from-scratch trainer; encoder collapse confound | `requires_grad=False` on `E`; no optimizer group; forward under `no_grad` |
| Target branch always stop-grad + EMA (**bottleneck EMA**) | Target chases predictor; collapse or trivial solutions | `as_target()` on `B_EMA`/`E` target outputs; no optimizer group for `B_EMA` |
| `c_t` bandwidth << `e_t` | Abstract state copies detailed latent; hierarchy fake | `N_c=32, D_c=256` vs `N_ctx=1024, D_e=1024`; monitor effective rank + cross-video cosine |
| `F_e` tested with shuffled/zero `c` | Decorative hierarchy passes undetected | `shuffled_c_test` in diagnostics; hard gate §9.4 |
| `L_e` must not backprop through `c_hat` into `F_c` | Fine loss turns `c` into texture carrier | `c_cond = as_target(c_hat)`; autograd test in Phase 2 |
| Frame decoder after latent learning; no encoder updates from frame loss | System becomes normal video generator | `requires_grad=False` on latent stack in Stage 4; `as_target(e_hat)` into D |
| Variance floor on `c_t` (no SIGReg/VICReg/covariance) | Constant/collapsed `c_t` passes undetected | `L_var` active Stage 1+; `λ_var=0.10`; **do not** add SIGReg/covariance |
| No horizontal/temporal flip on SSv2 | Label leakage / wrong semantics | Assert augmentations in `data.py` |
| Independent τ_c, τ_e | (Weaker training signal if tied) | Sample separately per example |

---

## 11. Design rationale

### Why two latents?

A single latent either carries everything (no compression — bypassable) or collapses under pressure. Splitting into **abstract** `c` (32×256) and **detailed** `e` (256×384 context / 64×384 target) forces future-relevant structure into a low-bandwidth bottleneck while preserving local appearance in `e`.

### Why flow matching, not endpoint regression?

Flow matching provides a **continuous generative path** from noise to target, conditioned on context. It matches the SIGReg Gaussian prior (noise is N(0,I)). Direct MSE on endpoints is harder to optimize and less compatible with the later frame generator also trained via flow matching.

### Why context/target symmetry (v0.2)?

Both branches use the **same frozen encoder** on length-`T` clips, so context and target share
geometry (`1024` tokens, dim `1024`). The target is the future clip `x_{≤t+k}`. The only difference is
the bottleneck (`B` online vs `B_EMA` target). Stability comes from the EMA bottleneck + stop-grad,
following **I-JEPA / V-JEPA** practice — not from a context/target masking asymmetry.

### Why a variance floor, not SIGReg/VICReg (v0.2)?

With a **frozen** encoder, only `c_t` (the bottleneck output) can collapse, and the EMA target +
flow objective already do most of the anti-collapse work. So the supervisor specified the **simplest
possible** guardrail — a per-dimension variance floor on `c_t` — and explicitly **no SIGReg, no full
VICReg, no covariance loss.** The variance floor only prevents constant `c_t`; the flow objective
learns the semantics. (SIGReg was the v0.1 choice; superseded — see §14 #30.)

### Why frame generator last?

Training D jointly with the encoder lets pixel reconstruction **shortcut** the hierarchy — D can ignore `e_hat` if it learns its own appearance model. Staged training verifies latents first via bypass tests, then adds rendering.

### Why EMA targets?

Stable prediction targets across steps. Without EMA, the target moves every gradient step and the online branch chases itself.

---

## 12. Literature grounding

| Architectural piece | Primary reference |
|---|---|
| Predicting representations, not pixels | I-JEPA (arXiv:2301.08243) |
| Latent video prediction, tubelets | V-JEPA (ICLR 2024) |
| Flow matching / rectified flow | Flow Matching for Generative Modeling (arXiv:2210.02747); Liu et al. rectified flow |
| Latent-space generation, cross-attention conditioning | Latent Diffusion Models (arXiv:2112.10752) |
| adaLN-Zero time conditioning | DiT (Peebles & Xie, 2022) |
| Variance-floor collapse prevention (v0.2) | VICReg variance term only (Bardes et al.) — not full VICReg/SIGReg |
| Frozen pretrained video encoder (v0.2) | V-JEPA 2 (arXiv:2506.09985); V-JEPA (ICLR 2024) |
| ConvNeXt-style mixing | ConvNeXt V2 |
| Perceiver-style learned queries | Perceiver IO (bottleneck 32 queries) |
| Dataset | Something-Something V2 (~220k videos, interaction-heavy) |
| Frozen image VAE | `stabilityai/sd-vae-ft-mse` (LDM ecosystem) |

The **distinctive design choice** of this project is the two-level coarse-to-fine latent transition with explicit bypass tests — not any single component above.

---

## 13. Dataset notes (Something-Something V2)

### Why SSv2 for v0

Action and object-interaction heavy; **temporal order matters** (left/right, moving toward/away). Better stress test than appearance-dominated datasets where static frames partially solve the task.

### Format and loading

- Videos: `.webm`, ~12 fps, variable length.
- Labels: `{video_id: template_string}` — 174 classes.
- Loader: **decord**, CPU decode per worker (not main process).
- Clip (v0.2): an **8-frame context** window ending at `t` and an **8-frame target** window ending at
  `t+k`, both at stride 2. For Phases 1–3, `k` is a single fixed horizon; Phase 4 samples
  `k∈{4,8,16,32}`. The video must be long enough to supply both windows; skip/clip-pad short videos
  deterministically and log counts.

### Preprocessing

| Setting | Value |
|---|---|
| Pixel normalization | **Encoder's processor normalization** (V-JEPA); VAE [-1,1] only in Stage 4 |
| Train resize/crop | shorter side → 256, random crop 256×256 |
| Eval resize/crop | shorter side → 256, center crop 256×256 |
| Color jitter | brightness/contrast/saturation = 0.4, hue = 0 |
| Horizontal flip | **OFF** (labels direction-sensitive) |
| Temporal flip | **OFF** |
| Rotation | **OFF** |

### RunPod data layout

Full volume tree (current state vs target): [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../SETUPS/VOLUME_LAYOUT.md).

```
/workspace/data/ssv2/{train,validation,labels.json}      ← full dataset (symlinks)
/workspace/data/ssv2_tiny/{train,validation,manifest.json}  ← smoke subset (~4k/348)
/workspace/ssv2_raw/                                     ← raw .webm (read-only)
```

Training CLI: `--data ssv2_tiny` (default, 4–5 hr smoke) or `--data ssv2` (full run). Override root: `JEPA_DATA_ROOT`.

### SSv2-tiny subset spec

Stratified sample: ~23 train + ~2 val videos per class, seed 42. Created by `make_subset.py` — symlinks only, no re-encoding. See `AGENT_FILES/PHASES/PHASE_1.md` §5.

---

## 14. Resolved decisions log

Every choice below was **locked** after brief review + external research + implementer approval ("go with recommendations"). No open alternatives remain.

| # | Topic | Decision | Reasoning |
|---|---|---|---|
| 1 | Bottleneck ConvNeXt depth | **2 blocks** | Enough spatial mixing without over-parameterizing v0 |
| 2 | Bottleneck query slots | **32 pure learned embeddings**, no input conditioning | Matches Perceiver IO; simpler than pooled conditioning |
| 3 | ConvNeXt mixing geometry | **Per-frame 2D** on 8×8 grids (context: 4 grids; target: 1 grid) | Structural parity context/target; avoids 3D ambiguity |
| 4 | F_c conditioning on `c_t` | **Concat** `[c_t \|\| z_c]` → 64 tokens, self-attention, read first 32 | Cheap at 64 tokens; fewer params than cross-attn |
| 5 | F_e conditioning | **Single cross-attn** to `concat(e_t, proj(c_cond))` | One memory; attention learns routing (DiT multi-modal pattern) |
| 6 | D dim / heads | **dim 512, 8 heads, MLP ratio 4** | Pixel-level generation harder than latent; ~38M params |
| 7 | `c_hat` at training | **One-step Euler:** `c_hat = z_c + (1-τ_c)·u_c_hat`, detached for F_e | Exact under rectified flow; cheap vs multi-step |
| 8 | `e_hat` at inference | **4-step Heun ODE** for F_c and F_e | Better quality at deploy; training uses one-step where noted |
| 9 | τ sampling | **Independent** `τ_c`, `τ_e` per example | Richer flow training signal |
| 10 | Condition dropout | **10% joint** replacement with learned null embeddings | Brief §7; CFG-style robustness |
| 11 | Pixel normalization | **[-1, 1]** | Matches SD VAE input range for Stage 4 consistency |
| 12 | Frame stride | **2** | ~0.83 s context; motion visible, still predictable |
| 13 | Frame VAE | **`stabilityai/sd-vae-ft-mse`** via diffusers | 8× downsample, 4 channels, LDM standard |
| 14 | Position embeddings | **3D sin-cos**, factorized T+H+W; target temporal index = 4 | V-JEPA v1 convention; coherent context/target coords |
| 15 | Tubelet dropout × pos emb | Drop **after** adding pos emb | MAE/V-JEPA standard |
| 16 | Time conditioning | **adaLN-Zero** in F_c, F_e, D | DiT standard; identity init at step 0 |
| 17 | SIGReg M / knots | **1024 / 17** | le-wm reference; insensitive per LeJEPA |
| 18 | Step budget | **30k / 25k / 50k / 45k** = 150k | Stage 4 needs ≥45k; Stage 3 largest (hardest) |
| 19 | Stage 3 ramp | **5k linear** steps at start of Stage 3 | Smooth teacher → predicted transition |
| 20 | EMA schedule S | **105,000** (latent stages sum) | No EMA updates in Stage 4 |
| 21 | LR warmup | **10k** latent (over 105k); **3k** Stage 4 (over 45k) | ~7–10% warmup fraction per phase |
| 22 | SIGReg from Stage 1 | **Yes**, both `e_t` and `c_t` | Prevent e_t drift before F_e exists |
| 23 | Diagnostic thresholds | See §2.6 | Copy/mean baselines, shuffled-c, rank floors |
| 24 | Implementation phases | **3 phases** = Stages 0+1 / 2+3 / 4 | Vertical slices; no scaffolding-only phases |
| 25 | RunPod layout | Code in repo; data/checkpoints siblings under `/workspace/` | Clone-and-run without nested data in repo |
| 26 | Default dataset CLI | **`ssv2_tiny`** | Safe smoke default; opt into full `ssv2` |

### v0.2 decisions (supervisor update — supersede conflicting earlier rows)

| # | Topic | Decision | Reasoning |
|---|---|---|---|
| 27 | Encoder | **Frozen pretrained V-JEPA 2 ViT-L/16** (dim 1024), shared both branches | Supervisor: frozen world-model ViT; SSv2 SOTA family; isolates the hierarchy. Supersedes #(encoder-from-scratch) |
| 28 | EMA scope | **Bottleneck only** (`B_EMA`); no `E_bar`; no encoder LR | Encoder frozen → nothing to EMA on the encoder. Supersedes §7 (both-branch EMA) |
| 29 | Tubelet dropout | **Removed** from encoder input; optional masking inside bottleneck later | Frozen encoder never saw dropped tokens. Supersedes #15 / Brief §7 (40%) |
| 30 | Collapse prevention | **Variance floor on `c_t` only**, `λ_var=0.10`, std target 1.0; **no SIGReg/VICReg/covariance** | Supervisor directive; minimal machinery. Supersedes #17, #22, §5.3 (SIGReg) |
| 31 | Prediction target | **Clip-level `c⁺_{t+k}=B_EMA(E(x_{≤t+k}))`** (length-`T` window ending at `t+k`) | Supervisor formula; matches encoder's clip pretraining; enables multi-horizon |
| 32 | Resolution / frames | **256×256, 8 context frames** (tubelet-2 → `N_ctx=1024`) | Balance feature richness vs cost; 4 frames gave too few temporal tokens (decided with human) |
| 33 | Input normalization | **Encoder's processor normalization** (not [-1,1]); VAE [-1,1] separate in Stage 4 | Must match the pretrained encoder. Supersedes #11 for the encoder path |
| 34 | Multi-horizon | **Phase 4 (deferred)**: `k∈{4,8,16,32}`, probs `{.30,.30,.25,.15}`, learned `h_k` | Land one change at a time; see `../PHASES/PHASE_4.md` |
| 35 | Phase 2/3 detailed target geometry | **OPEN** — recommended default: future-clip `e_{t+k}`; generator renders frame `t+k` | Supervisor specified coarse path only; confirm before Phase 2 code |

---
