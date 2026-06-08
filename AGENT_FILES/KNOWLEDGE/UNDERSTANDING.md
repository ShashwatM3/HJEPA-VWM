# UNDERSTANDING.md
## Hierarchical JEPA-Flow Video World Model — Comprehension Reference

> **Purpose.** This is an agent-facing comprehension reference for the architecture defined in [`hierarchical_jepa_flow_architecture_brief.pdf`](hierarchical_jepa_flow_architecture_brief.pdf) (same directory). It expands every component of the brief to the level of detail an implementation agent needs to write correct code: explicit shape contracts, exact gradient flow, exact training-stage semantics, exact constants. **There are no unresolved questions in this document.** Every choice not explicitly fixed by the brief has been resolved — either by external research with citation, or by deliberate decision recorded in §14. Where alternatives existed, one was chosen and the others discarded.
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

We are training a **video world model** whose internal representation is a **hierarchy of two latents**: an abstract latent `c_t` carrying the future-relevant structure of a short context clip, and a detailed latent `e_t` carrying texture and local visual information. The model learns by predicting **future latents**, not future pixels: from `c_t` it predicts `c_plus` (the future abstract state), and from `e_t` plus a coarse condition it predicts `e_plus` (the future detailed state). Both predictions are performed by **flow-matching networks** that learn a continuous-time velocity field transforming Gaussian noise into the future target latent. Future targets `e_plus` and `c_plus` come from an **EMA target encoder** — a slow-moving exponential-moving-average copy of the online encoder — and are always stop-gradient. **The frame decoder is a separate, later-stage module trained only after the latent world model is verified working; it never updates the encoder.** The single most important property of this design — the property all the bypass tests in §9 exist to verify — is that the abstract latent `c_t` must carry **real predictive signal** and must not collapse, copy, or be bypassed by the detailed latent or by the frame decoder.

This is **not** a video diffusion model. A normal video diffusion model trains a denoiser directly on pixel-space or VAE-latent-space targets; nothing forces it to develop a compressed predictive state. Here, the compressed state is the entire point, and the generative components (the flows, the frame decoder) exist only to teach and to render it.

---

## 2. Symbol and shape table

These are the canonical names and shapes used throughout the codebase. **Any function that produces or consumes any of these tensors must have a docstring shape contract matching this table exactly.** Batch dimension `B` is omitted from the "shape" column but is always present in code.

### 2.1 Inputs

| Symbol | What it is | Shape (without batch) | dtype | Origin |
|---|---|---|---|---|
| `X_ctx` | Context clip: the 4 frames immediately preceding the prediction target | `(T=4, C=3, H=128, W=128)` | float32 from loader, bf16 after AMP cast | Dataloader |
| `y` | Future frame to be predicted (single frame at time `t+1` after the context) | `(C=3, H=128, W=128)` | float32 → bf16 | Dataloader |

Pixel values are normalized to **[-1, 1]** (see §13 and §14 #11).

### 2.2 Latents produced by the encoder/bottleneck

| Symbol | What it is | Shape | dtype | Produced by | Branch |
|---|---|---|---|---|---|
| `e_t` | Detailed latent of the context clip | `(N_ctx=256, D_e=384)` | bf16 | Online encoder `E` | online (trainable) |
| `c_t` | Abstract latent of the context clip | `(N_c=32, D_c=256)` | bf16 | Online bottleneck `B` | online (trainable) |
| `e_plus` | Detailed latent of the future frame | `(N_tgt=64, D_e=384)` | bf16, **stop-grad** | EMA target encoder `E_bar` | target (no grad) |
| `c_plus` | Abstract latent of the future frame | `(N_c=32, D_c=256)` | bf16, **stop-grad** | EMA bottleneck `B_bar` | target (no grad) |

Token counts derived from the patch geometry:
- `N_ctx = 4 frames × (128 / 16)² = 4 × 64 = 256` — context patched with a 1×16×16 tubelet (per-frame 16×16 patches, no temporal merging).
- `N_tgt = 1 frame × (128 / 16)² = 1 × 64 = 64` — target is a single frame.
- `N_c = 32` — fixed by the bottleneck's 32 learned query slots, regardless of input length.

During training with tubelet dropout (§3.2.1), `e_t` has a variable post-dropout token count `N_ctx_post ≈ 154` (60% of 256, rounded per-batch); during eval the full 256 is used.

### 2.3 Predictions produced by the flow networks

| Symbol | What it is | Shape | dtype | Produced by | Notes |
|---|---|---|---|---|---|
| `z_c` | Noised version of `c_plus` at flow-time `τ_c` | `(N_c=32, D_c=256)` | bf16 | `(1-τ_c)·ε_c + τ_c·c_plus` | `ε_c ~ N(0, I)`, `τ_c ~ U(0,1)` |
| `u_c` | Ground-truth coarse velocity | `(N_c=32, D_c=256)` | bf16 | `c_plus - ε_c` | Constant along trajectory (rectified flow) |
| `u_c_hat` | Predicted coarse velocity | `(N_c=32, D_c=256)` | bf16 | `F_c(z_c, τ_c, c_t)` | Target of L_c |
| `c_hat` | One-step prediction of `c_plus`, used as coarse condition for F_e in Stage 3 | `(N_c=32, D_c=256)` | bf16, **stop-grad** when fed to F_e | `(z_c + (1 − τ_c) · u_c_hat).detach()` | One-step Euler — see §14 #5 |
| `z_e` | Noised version of `e_plus` at flow-time `τ_e` | `(N_tgt=64, D_e=384)` | bf16 | `(1-τ_e)·ε_e + τ_e·e_plus` | `ε_e ~ N(0, I)`, `τ_e ~ U(0,1)`, independently sampled from `τ_c` |
| `u_e` | Ground-truth fine velocity | `(N_tgt=64, D_e=384)` | bf16 | `e_plus - ε_e` |  |
| `u_e_hat` | Predicted fine velocity | `(N_tgt=64, D_e=384)` | bf16 | `F_e(z_e, τ_e, e_t, c_cond)` | `c_cond` is `c_plus` (Stage 2) or `stopgrad(c_hat)` (Stage 3) |
| `e_hat` | Predicted future detailed latent; conditioning into D in Stage 4 | `(N_tgt=64, D_e=384)` | bf16, **stop-grad** when fed to D | Full ODE rollout of F_e (4 Heun steps) at inference; one-step at Stage 4 training-time | See §14 #5 |

### 2.4 Frame-generator-stage quantities (Stage 4 only)

| Symbol | What it is | Shape | Notes |
|---|---|---|---|
| `a_y` | VAE-encoded future frame, patched for D | `(N_vae=64, D_vae_token=16)` | Underlying VAE latent is `(4, 16, 16)`; 2×2 spatial patching gives 8×8=64 tokens with 4×2×2=16 dims each |
| `z_x` | Noised patched VAE latent at flow-time `τ_x` | `(N_vae=64, D_vae_token=16)` | `(1-τ_x)·ε_x + τ_x·a_y` |
| `u_x` | Ground-truth velocity in patched VAE latent space | `(N_vae=64, D_vae_token=16)` | `a_y - ε_x` |
| `u_x_hat` | Predicted velocity | `(N_vae=64, D_vae_token=16)` | `D(z_x, τ_x, stopgrad(e_hat))`, after D's own unpatchify-projection |

The VAE used is `stabilityai/sd-vae-ft-mse` (frozen). For a 128×128 RGB input it produces a 4×16×16 latent. D's internal model dim is 512; it patchifies the 4×16×16 latent to 8×8=64 tokens of dim 16, projects to 512 for its transformer blocks, then projects back to dim 16 and unpatchifies.

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
| Number of context frames `T` | 4 | Context clip |
| Frame spatial size `H × W` | 128 × 128 | All frames |
| Patch size (T × H × W) | 1 × 16 × 16 | All patchifiers |
| `N_ctx` (context tokens) | 256 | Output of context patchifier |
| `N_tgt` (target tokens) | 64 | Output of target patchifier |
| `N_c` (abstract tokens) | 32 | Bottleneck query slot count |
| `D_e` (detailed latent dim) | 384 | Encoder, F_e, target encoder |
| `D_c` (abstract latent dim) | 256 | Bottleneck output, F_c |
| Encoder depth | 12 | Online + EMA target encoder |
| Encoder heads | 6 | Online + EMA target encoder |
| Encoder MLP ratio | 4 | Online + EMA target encoder |
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
| VAE downsampling | 8× | Yields 16×16 latent for 128×128 input |
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
| LR (encoder E, from scratch) | 2e-4 | Brief §7 |
| LR (bottleneck B) | 2e-4 | Brief §7 |
| LR (F_c) | 4e-4 | Brief §7 |
| LR (F_e) | 4e-4 | Brief §7 |
| LR (frame generator D) | 2e-4 | Brief §7 |
| LR schedule (latent stages 1–3 combined) | 10k warmup, cosine decay over remaining 95k | Brief §7 + §14 #7 |
| LR schedule (Stage 4) | 3k warmup, cosine decay over remaining 42k | §14 #7 |
| Tubelet dropout (context only) | 40% | Brief §7 |
| Condition dropout (flows) | 10% | Brief §7 |
| EMA momentum `m` start | 0.996 | Brief §3 |
| EMA momentum `m` end | 0.9999 | Brief §3 |
| EMA cosine schedule denominator `S` | 105,000 (sum of latent stages) | §14 #9 |

#### Losses

| Constant | Value | Source |
|---|---|---|
| `λ_fine` (L_e weight in total loss) | 1.0 | Brief §4.3 |
| `λ_e_reg` (SIGReg(e) weight) | 0.02 | Brief §4.3 |
| `λ_c_reg` (SIGReg(c) weight) | 0.10 | Brief §4.3 |
| SIGReg `M` (random projections) | 1024 | `lucas-maes/le-wm` reference impl |
| SIGReg `knots` (integration knots) | 17 | `lucas-maes/le-wm` reference impl |

#### Diagnostic thresholds (§9)

| Threshold | Value |
|---|---|
| F_c vs copy-baseline (val L_c ratio) after Stage 1 step 10k | ≤ 0.70 |
| F_c vs batch-mean baseline (val L_c ratio) after Stage 1 step 10k | ≤ 0.50 |
| Shuffled-c test (L_e ratio shuffled/real) by end Stage 2 | ≥ 1.5 |
| Shuffled-c test by end Stage 3 | ≥ 2.0 |
| Per-dim std collapse warning | >15% of dims with std < 0.1× median |
| Per-dim std collapse hard stop | >30% of dims |
| Effective rank `c_t` (D=256) — healthy | > 60 |
| Effective rank `c_t` — concerning | < 20 |
| Effective rank `c_t` — hard stop | < 5 |
| Effective rank `e_t` (D=384) — healthy | > 90 |
| Effective rank `e_t` — concerning | < 30 |
| Effective rank `e_t` — hard stop | < 8 |

#### Data

| Constant | Value | Source |
|---|---|---|
| Frame stride (within a clip) | 2 | §14 #12 |
| Pixel normalization | [-1, 1] | §14 #11 |
| Resize policy (train) | shorter-side to 128, then random-crop to 128×128 | §14 R9 |
| Resize policy (eval) | shorter-side to 128, then center-crop to 128×128 | §14 R9 |
| Horizontal flip | OFF | §14 R10 |
| Temporal flip | OFF | §14 R10 |
| Color jitter | brightness=0.4, contrast=0.4, saturation=0.4 (no hue) | §14 R10 |

---

## 3. The architecture in full

For each module: input, output, parameter budget, behavioral requirements. Every implementation choice is fixed in this section — there are no "either/or"s.

### 3.1 Patchification / tokenization

There are **two patchifiers**, both with patch shape 1×16×16 (temporal × height × width). Both share the same `Conv3d` projection weights — they are the same operator applied to different input lengths.

**Context patchifier (input to online encoder):**
1. Apply `Conv3d` with kernel `(1, 16, 16)`, stride `(1, 16, 16)`, out-channels `D_e=384`. Input `(B, 3, 4, 128, 128)` → output `(B, 384, 4, 8, 8)`.
2. Reshape to `(B, 256, 384)` (flatten spatial-temporal).
3. **Add absolute 3D sin-cos position embeddings** factorized along (T, H, W), then summed (V-JEPA v1 convention, see §12). Implementation: precompute three 1D sin-cos tables of lengths 4 (temporal), 8 (height), 8 (width), each of dim 384; for token at index `(t, h, w)` add `pe_t[t] + pe_h[h] + pe_w[w]`. These are constants, not learned.
4. Apply tubelet dropout (random 40% drop, see §3.2.1).

**Target patchifier (input to target encoder):**
1. Apply the same `Conv3d`. Input `(B, 3, 1, 128, 128)` → output `(B, 384, 1, 8, 8)`.
2. Reshape to `(B, 64, 384)`.
3. **Add the same 3D sin-cos position embeddings, but with the temporal index set to `T=4`** — i.e., the target is positioned at "the frame immediately after the 4-frame context window." Spatial indices match the context's spatial indices. (The temporal table must therefore have length ≥ 5 in the precomputation; use length 8 for headroom.)
4. **No tubelet dropout on the target side, ever.**

### 3.2 Online encoder `E` — VideoViT-Small

Standard ViT-Small dimensions: **depth 12, dim 384, 6 attention heads, MLP ratio 4, no [CLS] token, LayerNorm (pre-norm)**.

- **Input:** context tokens of shape `(N_ctx_post_dropout ≈ 154, 384)` during training, `(256, 384)` during eval.
- **Output:** `e_t` of the same shape as input (one output token per input token).
- **Attention:** full self-attention over all tokens. At ≤256 tokens this is cheap.
- **Parameter budget:** ≈ 22M (matches ViT-S).
- **Norm style:** pre-norm.

#### 3.2.1 Tubelet dropout

Applied only to the context side, with `p_tub = 0.40`. Implementation:

1. After position embeddings are added (so dropped tokens still carry their position info on the *remaining* tokens), select a random 60% of the 256 tokens to keep (per-example mask, sampled fresh each step).
2. Pass only the kept tokens through `E`.
3. **Output `e_t` therefore has a variable token count during training** (~154 expected, varies). The flows F_c and F_e use cross-attention with `e_t` as keys/values, so variable length is naturally supported.
4. Save the binary kept-mask so the bottleneck (§3.3) can reconstruct the spatial grid for ConvNeXt mixing.

The EMA target encoder `E_bar` never has tubelet dropout — it always processes all 64 target tokens.

### 3.3 Bottleneck `B`

Compresses `e_t` (~154 context tokens of dim 384, post-dropout) into `c_t` (32 abstract tokens of dim 256). Or compresses `e_plus` (64 target tokens of dim 384) into `c_plus` (32 tokens of dim 256) on the EMA branch — same architecture, applied to a different input length.

**Structure (fixed):**

1. **Spatial-grid reconstruction + per-frame 2D ConvNeXt mixing (2 blocks).**
   - **Context side:** using the kept-mask from §3.2.1, scatter the ~154 surviving `e_t` tokens back into a `(B, 4, 8, 8, 384)` grid, with dropped slots filled with zero vectors. Reshape to `(B·4, 8, 8, 384)` so each of the 4 frame grids is a separate "image" for the conv.
   - **Target side:** reshape `e_plus` directly to `(B, 1, 8, 8, 384)` → `(B, 8, 8, 384)`.
   - Apply 2 ConvNeXt-V2-style blocks to the spatial grid. Each block: `Conv2d` 7×7 depthwise with channel dim 384 → LayerNorm → pointwise linear 384→4·384 → GELU → pointwise linear 4·384→384, with a residual connection. Weights shared between the two blocks's structure but each block has its own parameters. The same B is applied to all 4 context frame grids (weights shared across the 4) and to the 1 target grid.
   - After the ConvNeXt blocks, flatten spatial back to tokens: context becomes `(B, 256, 384)` (including the zero-filled dropped slots), target stays `(B, 64, 384)`.
   - **Drop the zero-filled positions back out** for the context side (using the kept-mask) so the cross-attention sees only real tokens: back to `(B, ~154, 384)`.

2. **Cross-attention to 32 learned query embeddings.** A single cross-attention layer:
   - Queries: 32 learnable embeddings of dim 256 (parameters of B, no per-input conditioning), broadcast to batch.
   - Keys, values: post-mixer tokens projected from dim 384 to dim 256 by a linear layer, then used as `(N_input, 256)` keys/values.
   - 8 attention heads.
   - Output: 32 tokens of dim 256.

3. **Output MLP block.** One transformer-style block on the 32 output tokens: LayerNorm → linear 256→1024 → GELU → linear 1024→256, with residual. Final LayerNorm.

**Parameter budget:** ≈ 3M.

**Outputs:**
- Online branch: `c_t = B(e_t, kept_mask_ctx)` — `(32, 256)`.
- Target branch: `c_plus = B_bar(e_plus, None)` — `(32, 256)`, then `.detach()`.

### 3.4 EMA target branch (`E_bar`, `B_bar`)

Structurally identical to the online branch. Updated only by EMA (§7), never by backprop.

- `E_bar`: takes `(64, 384)` target tokens (single frame, no dropout). Outputs `e_plus` of shape `(64, 384)`.
- `B_bar`: takes `e_plus`. Outputs `c_plus` of shape `(32, 256)`.

**Implementation requirements:**
- Separate `nn.Module` instances (not weight-sharing — they are independent parameter copies that the EMA rule keeps approximately aligned).
- `param.requires_grad = False` on every parameter.
- Permanently in `eval()` mode (deterministic — no dropout). Since this architecture uses only LayerNorm (no BatchNorm) this is largely a no-op for normalization layers; but it does ensure tubelet dropout in `E_bar`'s patchifier is disabled (which is already the design rule, but `eval()` reinforces it).
- Initialized at Stage 0 by copying `E.state_dict()` into `E_bar` and `B.state_dict()` into `B_bar`.

### 3.5 Asymmetry between context and target representations

The target encoder processes the **full unmasked future frame**. There is no target-side masking or dropout. All 64 target tokens become prediction targets. The asymmetry between context (multi-frame, 256 tokens, with tubelet dropout) and target (single frame, 64 tokens, no dropout) is the architectural expression of the I-JEPA design principle: the target encoder runs on a fully visible input to produce semantically rich representations that the predictor must regress against.

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
- `X_ctx`: `(4, 3, 128, 128)` — context frames, pixels in **[-1, 1]**
- `y`: `(3, 128, 128)` — future frame, **[-1, 1]**

### Step 1 — Context patchify + tubelet dropout

```
tokens_ctx, kept_mask = patchify_context(X_ctx)
# tokens_ctx: (N_ctx_post ≈ 154, 384) after 40% tubelet dropout
# kept_mask: (256,) binary, for bottleneck grid reconstruction
```

Position embeddings added **before** dropout; surviving tokens retain their spatial-temporal coordinates.

### Step 2 — Online encoder

```
e_t = E(tokens_ctx)
# e_t: (N_ctx_post ≈ 154, 384)
```

### Step 3 — Online bottleneck

```
c_t = B(e_t, kept_mask)
# c_t: (32, 256)
```

### Step 4 — Target branch (no grad)

```
with torch.no_grad():
    tokens_tgt = patchify_target(y)           # (64, 384), no dropout
    e_plus = E_bar(tokens_tgt)                # (64, 384)
    c_plus = as_target(B_bar(e_plus))         # (32, 256), detached
    e_plus = as_target(e_plus)                # (64, 384), detached
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
L_reg_e = SIGReg(e_t)    # online detailed latent only
L_reg_c = SIGReg(c_t)    # online abstract latent only
L = L_c + 1.0 * L_e + 0.02 * L_reg_e + 0.10 * L_reg_c
```

See §5 for gradient routing.

### Step 13 — Backward + clip + optimizer step

```
L.backward()
clip_grad_norm_(all_trainable_params, 1.0)
optimizer.step()
optimizer.zero_grad()
```

Trainable in Stage 3: `E`, `B`, `F_c`, `F_e`. Not trainable: `E_bar`, `B_bar`, detached targets.

**Verify:** `c_hat` path into `F_e` uses `as_target(c_hat)` so **no gradient from L_e reaches F_c**.

### Step 14 — EMA update (latent stages only)

```
m = ema_cosine(step, start=0.996, end=0.9999, total=105_000)
for p_online, p_ema in zip(E.params, E_bar.params):
    p_ema.lerp_(p_online, 1 - m)
# Same for B → B_bar
```

No EMA update during Stage 4 (encoder frozen).

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
| `E`, `B`, `F_c` | `E_bar`, `B_bar`, `c_plus`, `e_plus` |

`c_t` is **not** stop-gradient on the conditioning path — the encoder must learn predictive abstract states.

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

### 5.3 SIGReg — Gaussian latent regularization

**Sketched Isotropic Gaussian Regularization** (LeJEPA / Balestriero & LeCun, arXiv:2511.08544). Reference implementation: `lucas-maes/le-wm`.

Applied to **online latents only**: `e_t` and `c_t` (brief §4.3 names `e_t` and `c_t` explicitly — not EMA branch outputs).

Mechanism (summary):
1. Flatten tokens per batch: `(B, N, D) → (B·N, D)` or pool — match reference impl structure.
2. Draw `M=1024` random 1D projection directions (Cramér-Wold).
3. Project latents; test each projection against N(0,1) via **Epps-Pulley** characteristic-function statistic.
4. Integrate over `knots=17` points with trapezoidal rule.
5. Return scalar loss.

Hyperparameters `M` and `knots` are **not** sensitive per LeJEPA ablations — only λ weights matter for tuning.

```
L_reg = λ_e_reg · SIGReg(e_t) + λ_c_reg · SIGReg(c_t)
λ_e_reg = 0.02,  λ_c_reg = 0.10
```

SIGReg is active from **Stage 1 step 0** (both latents), even before `F_e` exists — prevents `e_t` drift that would complicate Stage 2 entry.

**Why SIGReg specifically:** flow networks start from `N(0,I)` noise; regularizing latents toward isotropic Gaussian aligns the representation distribution with the generative source distribution.

### 5.4 Total latent objective (Stages 1–3)

```
L_latent = L_c + λ_fine · L_e + λ_e_reg · SIGReg(e_t) + λ_c_reg · SIGReg(c_t)
λ_fine = 1.0
```

**Stage 1 only:** `L_latent = L_c + λ_e_reg · SIGReg(e_t) + λ_c_reg · SIGReg(c_t)` (no L_e).

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
| 1 | `e_plus` | **Always yes** | After `E_bar` output | EMA target — predicted, never optimized |
| 2 | `c_plus` | **Always yes** | After `B_bar` output | Same |
| 3 | `c_hat` → `F_e` input | **Yes** | Stage 3+ before `c_cond` | Prevents L_e from backprop into F_c |
| 4 | `e_hat` → `D` input | **Yes** | Stage 4 always | Frame gen must not rewrite world model |
| 5 | `E_bar`, `B_bar` | **No backprop ever** | All stages | EMA update only |
| 6 | `e_t`, `c_t` on flow conditioning | **No** | Latent training | Encoder/bottleneck must learn predictive structure |
| 7 | `u_c_hat` → `c_hat` → `F_e` | **Partial** | `c_hat` detached; `u_c_hat` not detached w.r.t. L_c | L_c trains F_c; L_e must not |
| 8 | VAE encode/decode | **Always yes** | Stage 4 | Frozen pretrained encoder |
| 9 | Latent stack in Stage 4 | **Frozen (no grad)** | `requires_grad=False` | Only D trains |

**Invariant:** Anything from `E_bar` or `B_bar` is detached. Nothing from `E` or `B` is detached on the path into losses as conditioning (except where explicitly noted for `c_hat` into `F_e`).

---

## 7. EMA: what, where, how, schedule

### What is EMA'd

Only the **target branch** copies: `E_bar` (target encoder) and `B_bar` (target bottleneck). Flow networks and frame generator are **not** EMA'd.

### Initialization (Stage 0)

At training start, before any optimizer step:

```
E_bar.load_state_dict(E.state_dict())
B_bar.load_state_dict(B.state_dict())
```

### Update rule (after each optimizer step, Stages 1–3)

```
m = ema_cosine(step, m_start=0.996, m_end=0.9999, S=105_000)
θ_ema ← m · θ_ema + (1 - m) · θ_online
```

Implemented as `p_ema.lerp_(p_online, 1 - m)` in PyTorch.

The schedule is **parametric** (cosine in step index), not tied to loss dynamics. `S = 105_000` = sum of latent training stages (30k + 25k + 50k). During Stage 4, EMA updates **do not run** (encoder frozen).

### Why EMA targets exist

The online encoder chases a moving target if trained against its own immediate outputs. The slow EMA branch provides **stable future representations** to predict, following I-JEPA / V-JEPA practice. Without EMA + stop-grad, representations collapse or chase the predictor every step.

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

Pass condition: dataloader loads batches; all modules construct; `E_bar`/`B_bar` initialized from online copies; one synthetic forward + backward + EMA update completes without NaN; bf16 AMP stable.

### Stage 1 — Coarse dynamics

**Train:** `E`, `B`, `F_c`. **Frozen:** `E_bar`, `B_bar` (no grad, EMA updated).

**Loss:** `L_c + λ_e_reg·SIGReg(e_t) + λ_c_reg·SIGReg(c_t)`.

**Pass condition:**
- `c_t` non-collapsed (effective rank > 60, std health).
- `F_c` beats **copy baseline** (val L_c ratio ≤ 0.70) and **batch-mean baseline** (≤ 0.50) after step ≥ 10k.

Copy baseline = predict future abstract state equals current `c_t`. Batch-mean = predict `c_plus` as mean of batch's `c_plus`.

### Stage 2 — Fine teacher-forcing

**Train:** `E`, `B`, `F_c`, `F_e`. **Condition:** `c_cond = c_plus`.

**Loss:** full `L_latent`.

**Pass condition:** `F_e` works with true stopped `c_plus`; shuffled-c ratio ≥ **1.5** by step 55k.

**Ablation:** zeroing `c_plus` in `F_e` must hurt `L_e` — if not, hierarchy is decorative.

### Stage 3 — Predicted-coarse fine training

**Train:** same modules. **Condition ramp:** linear mix from `c_plus` to `as_target(c_hat)` over first **5,000** steps of Stage 3 (steps 55k→60k), then 100% predicted-coarse.

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
| `E` | 2e-4 | (0.9, 0.95) | 0.05 |
| `B` | 2e-4 | (0.9, 0.95) | 0.05 |
| `F_c` | 4e-4 | (0.9, 0.95) | 0.05 |
| `F_e` | 4e-4 | (0.9, 0.95) | 0.05 |
| `D` | 2e-4 | (0.9, 0.95) | 0.05 |

Latent LR schedule: **10k warmup**, cosine decay over remaining **95k** steps (Stages 1–3 combined). Global batch **64** clips; gradient accumulation allowed if OOM.

---

## 9. Runtime tests and early-stop gates

Run every **500–1,000** steps on a **fixed validation batch** (cache at init). Thresholds from §2.6.

### 9.1 Latent std

**Measure:** per-dimension mean and std of `e_t` and `c_t` across val batch.

**Failure:** many dimensions → 0 or flatline.

| Level | Condition |
|---|---|
| Warning | > 15% of dims with std < 0.1× median std |
| Hard stop | > 30% of dims |

### 9.2 Effective rank

**Measure:** covariance effective rank of flattened `e_t` and `c_t`.

| Latent | Healthy | Concerning | Hard stop |
|---|---|---|---|
| `c_t` (D=256) | > 60 | < 20 | < 5 |
| `e_t` (D=384) | > 90 | < 30 | < 8 |

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
| Target branch always stop-grad + EMA | Target chases predictor; collapse or trivial solutions | `as_target()` on all `E_bar`/`B_bar` outputs; no optimizer group for EMA params |
| `c_t` bandwidth << `e_t` | Abstract state copies detailed latent; hierarchy fake | `N_c=32, D_c=256` vs `N_ctx=256, D_e=384`; monitor effective rank |
| `F_e` tested with shuffled/zero `c` | Decorative hierarchy passes undetected | `shuffled_c_test` in diagnostics; hard gate §9.4 |
| `L_e` must not backprop through `c_hat` into `F_c` | Fine loss turns `c` into texture carrier | `c_cond = as_target(c_hat)`; autograd test in Phase 2 |
| Frame decoder after latent learning; no encoder updates from frame loss | System becomes normal video generator | `requires_grad=False` on latent stack in Stage 4; `as_target(e_hat)` into D |
| SIGReg + tubelet dropout together | Collapse or shortcut learning | Both active Stage 1+; 40% context dropout only |
| No horizontal/temporal flip on SSv2 | Label leakage / wrong semantics | Assert augmentations in `data.py` |
| Independent τ_c, τ_e | (Weaker training signal if tied) | Sample separately per example |

---

## 11. Design rationale

### Why two latents?

A single latent either carries everything (no compression — bypassable) or collapses under pressure. Splitting into **abstract** `c` (32×256) and **detailed** `e` (256×384 context / 64×384 target) forces future-relevant structure into a low-bandwidth bottleneck while preserving local appearance in `e`.

### Why flow matching, not endpoint regression?

Flow matching provides a **continuous generative path** from noise to target, conditioned on context. It matches the SIGReg Gaussian prior (noise is N(0,I)). Direct MSE on endpoints is harder to optimize and less compatible with the later frame generator also trained via flow matching.

### Why context/target asymmetry?

Context: 4 frames, 256 tokens, tubelet dropout — forces inference from partial views. Target: 1 full frame, 64 tokens, no dropout — rich EMA targets following **I-JEPA** (target encoder sees full input).

### Why SIGReg, not VICReg?

Flow predictors start from Gaussian noise; isotropic Gaussian regularization aligns latent statistics with the generative source. VICReg adds variance/covariance tuning burden without matching the flow prior as directly (LeJEPA line).

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
| SIGReg / LeJEPA Gaussian regularization | LeJEPA (arXiv:2511.08544); ref impl `lucas-maes/le-wm` |
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
- Clip: 5 frames sampled with **stride 2** → ~0.83 s span; frames 0–3 context, frame 4 target.

### Preprocessing

| Setting | Value |
|---|---|
| Pixel normalization | **[-1, 1]** (`x = x/255 * 2 - 1` or equivalent) |
| Train resize/crop | shorter side → 128, random crop 128×128 |
| Eval resize/crop | shorter side → 128, center crop 128×128 |
| Color jitter | brightness/contrast/saturation = 0.4, hue = 0 |
| Horizontal flip | **OFF** (labels direction-sensitive) |
| Temporal flip | **OFF** |
| Rotation | **OFF** |

### RunPod data layout

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

---
