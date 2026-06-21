# BRIEF_V0_3.md — Hierarchical JEPA-Flow Video World Model

> **What this file is.** The **current** architecture brief. It supersedes
> [`hierarchical_jepa_flow_architecture_brief.pdf`](hierarchical_jepa_flow_architecture_brief.pdf),
> [`BRIEF_V0_1.md`](BRIEF_V0_1.md), and [`BRIEF_V0_2.md`](BRIEF_V0_2.md) wherever they differ.
> v0.1 is the original PDF baseline; v0.2 is the frozen-encoder supervisor update; **v0.3 folds in
> everything implemented and empirically validated in Phase 1 Stage 1** (init fixes, horizon, loss
> weights, diagnostics, run history).
>
> **Precedence.** This file wins on architecture intent and Phase 1 operating defaults. Numerical
> constants in code live in [`UNDERSTANDING.md`](UNDERSTANDING.md) §2.6 and `config.py`; if those
> drift from this brief, reconcile them explicitly.

---

## Changes vs v0.2 (read first)

| # | Topic | v0.2 | v0.3 (current) |
|---|---|---|---|
| 1 | Bottleneck query init | `randn × 0.02` (brief default) | **`orthogonal_` (unit-norm rows)** — baked in, not flag-gated |
| 2 | Bottleneck `out_mlp` last layer | PyTorch default | **Zero-init** (adaLN-Zero-style identity start) — baked in |
| 3 | Prediction horizon | Single fixed `k` (deferred multi-horizon) | **`horizon_k = 12`** original frames (CLI `--horizon-k`); k=4 was too easy |
| 4 | Variance floor weight | `λ_var = 0.1` | **`λ_var = 0.5`** empirically required; 0.1 loses to `L_flow` |
| 5 | Optional regularizers | “No VICReg/covariance” | **VICReg-C + slot-diversity exist in code, default off**; slot loss rejected empirically |
| 6 | Coarse flow conditioning | Mentions `h_k` (Phase 4) | **No `h_k` in code yet** — concat self-attn over `[z_τ ∥ c_t]`, fixed horizon only |
| 7 | Learning rates | Bottleneck 2e-4, flows 4e-4 | **Bottleneck 1e-4, coarse flow 2e-4** (post–Run 1 stability) |
| 8 | Grad clip / skip | Clip 1.0 | **Clip 0.5; skip optimizer step if pre-clip norm > 50** |
| 9 | Warmup | 10k steps (brief default) | **1,500 steps** |
| 10 | Diagnostics | std, eff-rank, baselines, cross-video cosine | **+ `c_slot_diversity_rank`, per-head `c_attn_entropy`**; slot metric informative, entropy metric weak |
| 11 | Phase 1 status | Spec only | **Stage 1 passing** on run `cerulean-snow-13` (see §13) |

**Unchanged from v0.2:** frozen V-JEPA 2 encoder, EMA on bottleneck only, clip-level targets, rectified
flow on `c_t`, stop-grad on all targets, two-level hierarchy roadmap (coarse → fine → frame gen),
bypass-test mindset, frame decoder after latent training.

---

## Plain-English architecture brief

**One-sentence proposition:** train a video world model that first predicts the next **abstract**
latent state, then (later) the next **detailed** latent state, and only after that trains a separate
generator to render frames.

The goal is not a normal pixel diffusion model. The goal is a **predictive latent hierarchy**: the
compressed state must carry future-relevant structure; the detailed latent carries texture and local
visual information.

**Key rule:** the frame decoder must not teach the encoder during main training. The encoder is
**pretrained and frozen**; the **bottleneck and flows** are taught by future latent prediction
against a stop-gradient EMA target branch.

---

## 1. The architecture in plain English

At time `t`, the model receives a short **context clip** (8 frames, stride 2). A **frozen pretrained
ViT** (V-JEPA 2 ViT-L/16) turns that clip into detailed tokens `e_t`. A trainable **bottleneck**
compresses `e_t` into abstract state `c_t` (32 tokens × 256 dim).

For horizon `k` (currently **12 original frames**), the **same frozen encoder** encodes a **target
clip** starting at `t+k`. An **EMA copy of the bottleneck** produces target abstract latent
`c_plus` (= `c⁺_{t+k}`). Targets are **stop-gradient**.

A **coarse flow** model `F_c` learns rectified-flow velocity from noise toward `c_plus`, conditioned
on `c_t`. Later stages add **fine flow** `F_e` (detailed latent) and **frame generator** `D`.

```
Data path
---------
context clip x_{<=t}     -> [FROZEN E]              -> e_t     (B, 1024, 1024)
e_t                      -> [TRAINABLE B]           -> c_t     (B, 32, 256)

target clip x_{<=t+k}    -> [SAME FROZEN E]         -> e_plus  (B, 1024, 1024)
e_plus                   -> [EMA B_EMA, stop-grad]  -> c_plus  (B, 32, 256)
```

```
Coarse prediction (Phase 1 — implemented)
-----------------------------------------
Sample z_0 ~ N(0,I), tau ~ U(0,1)
z_tau    = (1 - tau) * z_0 + tau * c_plus
v_target = c_plus - z_0
L_flow   = || F_c(z_tau, tau, c_t) - v_target ||^2

F_c: concat z_tau and c_t -> 64-token sequence; AdaLN-Zero blocks; tau embedding;
     output velocity for first 32 tokens. (No horizon embedding h_k yet.)
```

```
Future path (not yet implemented)
---------------------------------
c_t -> F_c -> c_hat  (inference: integrate velocity tau: 0 -> 1)
(e_t, c_plus) -> F_e -> e_hat   [teacher-forced, then stopgrad(c_hat)]
stopgrad(e_hat) -> D -> x_hat
```

**Clip geometry (SSv2, current defaults):** 8 sampled frames per clip → 4 tubelets × 256 spatial
tokens = 1024 encoder tokens. Context and target clips are the same length; target starts `k`
original frames later (`k=12` ⇒ slight overlap with context; `k≥16` ⇒ non-overlapping).

---

## 2. Recommended v0 dimensions (implemented)

| Component | Current v0.3 | Reason |
|---|---|---|
| Input | 8 context frames @ 256×256, stride 2; target clip same length, offset `k=12` | Real motion in tubelet-2 encoder; harder task than k=4 |
| Encoder E | Frozen V-JEPA 2 ViT-L/16, dim 1024, tubelet 2, patch 16 | Pretrained SSv2 SOTA family; shared both branches |
| Detailed latent e_t | 1024 × 1024 | Set by encoder |
| Bottleneck B | ConvNeXt ×2 on 4×16×16 grids + 32 query cross-attn; proj 1024→256 | Compression + local spatial context before pooling |
| Abstract latent c_t | 32 × 256 (~8k scalars vs ~1M in e_t) | Tight enough to prevent copying |
| Coarse flow F_c | 6 AdaLN-Zero blocks, dim 256, 8 heads | Predicts full (32,256) velocity field |
| Fine flow F_e | *Not built* | Phase 2 |
| Frame generator D | *Not built* | Phase 4 |

**Bottleneck init (non-negotiable defaults in code):**
- Queries: `nn.init.orthogonal_` — fixes near-uniform attention from tiny random queries.
- `out_mlp` last layer: zeros — identity start for residual MLP branch.

---

## 3. What teaches what?

**Encoder E:** nothing — frozen, `requires_grad=False`.

**Bottleneck B and coarse flow F_c:** gradients from `L_flow` and `L_var` (and optional regularizers
if enabled).

**Target branch:** `c_plus` from `B_EMA(E(target_clip))`, always stop-gradient. `B_EMA` updates only
by EMA, never by backprop.

```
B_EMA <- m * B_EMA + (1 - m) * B
m: start 0.996, cosine-ramp toward 0.9999
```

There is **no** target encoder `E_bar` — the same frozen `E` serves both branches.

---

## 4. Losses

### 4.1 Coarse rectified-flow loss (Phase 1 mainline)

```
z_0 ~ N(0, I)          shape (B, 32, 256)
tau ~ U(0, 1)
z_tau = (1 - tau) * z_0 + tau * c_plus
v_target = c_plus - z_0
L_flow = mean_square( F_c(z_tau, tau, c_t) - v_target )

Updates:     B, F_c
No updates:  E, B_EMA, c_plus
```

### 4.2 Variance floor on c_t (primary anti-collapse lever)

Flatten `c_t` per batch element; per-dimension std across batch; hinge at target 1.0:

```
L_var = (1/d) * sum_j max(0, 1.0 - Std(c_j))     d = 32 * 256
```

**Empirical finding:** at `λ_var = 0.1`, `c_std_mean` stuck ~0.45 and latents collapsed toward
video-independent codes (`c_cross_video_cosine` → 0.7–0.8). At **`λ_var = 0.5`**, std reaches
~1.0, cross-video cosine ~0.20, collapse stops. This is the current operating default.

```
L = L_flow + lambda_var * L_var
lambda_var = 0.5   (CLI: --lambda-var)
```

### 4.3 Optional regularizers (in code, default off)

| Term | Purpose | Default | Empirical note |
|---|---|---|---|
| `L_cov` (VICReg-C) | Decorrelate feature dims | `λ_cov = 0` | Logged always; enable via `--lambda-cov` if rank plateaus low after stable run |
| `L_slot` | Penalize redundant slots within video | `λ_slot = 0` | **Rejected as training objective** — Goodhart: improves slot metric while worsening video-specific information; centered version also caused grad explosions |

Do **not** treat slot-diversity or raw attention entropy as primary optimization targets without
checking `c_cross_video_cosine` and `c_effective_rank` together.

### 4.4 Fine flow and frame losses

Unchanged in intent from v0.2 — **not implemented in Phase 1**. See `BRIEF_V0_2.md` §4.2–4.4 for
spec; `L_e` must not backprop through `c_hat` into `F_c`; frame loss updates `D` only.

---

## 5. Stop-gradient rules

| Object | Stop-gradient? | Why |
|---|---|---|
| `c_plus`, `e_plus` | Always yes | EMA / frozen targets |
| Encoder `E` | Frozen always | Pretrained; never updated |
| `B_EMA` | No backprop ever | EMA only |
| `c_hat` into `F_e` | Yes (Stage 3) | Prevents fine loss corrupting coarse predictor |
| `e_hat` into `D` | Yes (Stage 4) | Frame gen must not rewrite world model |
| `B`, `c_t` on conditioning path | No during latent training | Must preserve predictive information |

---

## 6. Training schedule

| Stage | Train | Loss | Pass condition (Phase 1 focus) |
|---|---|---|---|
| 0. Setup | Init `B_EMA = B`; load + freeze `E` | None | Shapes, EMA, AMP, data loader stable |
| **1. Coarse dynamics** | **B, F_c** | **`L_flow + λ_var·L_var`** | `c_t` noncollapsed; `F_c` beats copy & batch-mean baselines |
| 2. Fine teacher forcing | B, F_c, F_e | + `L_e` | `F_e` works with stopped `c_plus` |
| 3. Predicted-coarse fine | B, F_c, F_e | + `L_e` with `stopgrad(c_hat)` | Shuffled `c_hat` hurts |
| 4. Frame generator | D only | `L_frame` | Shuffled `e_hat` fails |
| (Later) Multi-horizon | B, F_c + `h_k` | Multi-`k` sampling | One model, multiple horizons |

**Phase 1 Stage 1 acceptance (current bar):** variance healthy, cross-video cosine well below ~0.5,
effective rank not collapsing, `coarse_vs_copy_ratio` < 1 after warmup. Met on run `cerulean-snow-13`.

---

## 7. Optimizer and runtime defaults

| Setting | Current default |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.05 |
| Gradient clipping | Global norm **0.5** |
| Grad skip guard | Skip step if pre-clip norm **> 50** |
| Precision | bfloat16 (encoder forward under `no_grad`) |
| Batch | Global 64 |
| LR bottleneck | **1e-4** |
| LR coarse flow | **2e-4** |
| LR encoder | none (frozen) |
| Warmup | **1,500** steps, then cosine decay |
| Max steps (Phase 1) | 15,000 |
| Condition dropout | ~10% on `c_t` for `F_c` |
| Tubelet dropout | Removed (frozen encoder) |
| Input norm | V-JEPA ImageNet stats (not [-1,1]) |
| Horizon | **`horizon_k = 12`** (`--horizon-k`) |
| Variance weight | **`lambda_var = 0.5`** (`--lambda-var`) |

---

## 8. Dataset

**Something-Something V2** — ~220k short human-object interaction clips; temporal order matters.
Train at **256×256**, 8-frame context + 8-frame target, single fixed horizon first (multi-horizon
with `h_k` is Phase 4).

At batch 64, 15k steps ≈ one epoch over full train split — step-based training, not epoch-based.

---

## 9. Runtime tests and early-stop gates

Run on a fixed diagnostic batch every **500** steps (`diag_every`).

| Test | Metric | Investigate if |
|---|---|---|
| Latent std | `c_std_mean`, `c_std_median` | Stuck well below 1.0 while `λ_var` active |
| Cross-video similarity | **`c_cross_video_cosine`** | Rises toward ~0.7+ (video-independent collapse) |
| Effective rank | **`c_effective_rank`** | Falls sharply or plateaus very low (~5) |
| Slot diversity | `c_slot_diversity_rank` | Collapses toward ~1–2 *and* copy ratio degrades |
| Coarse baseline | `coarse_vs_copy_ratio`, `coarse_vs_batch_mean_ratio` | Stays >> 1 after warmup |
| Flow loss | `L_flow` | Diverges or flatlines with bad health metrics |
| Gradient health | `grad_norm`, `grad_skipped`, `grad_has_nan` | Frequent skips, 10³+ spikes, NaNs |

**Metric caveats:**
- `c_cross_video_cosine` is mean pairwise cosine among **random batch pairs** — detects
  video-agnostic collapse, not semantic clustering of similar actions.
- `c_attn_entropy` is weak (near 1.0 even when slots differ) — prefer `c_slot_diversity_rank`.
- `coarse_vs_copy_ratio` is only trustworthy when `copy_loss` is stable (latent not drifting in scale).

---

## 10. Implementation milestones

**Done (Phase 1 Stage 1):** frozen encoder load, bottleneck + EMA, coarse flow, variance floor,
diagnostics, training script, SSv2 loader, W&B logging, CLI overrides (`--horizon-k`, `--lambda-var`,
`--lambda-cov`, `--lambda-slot`).

**Next:**
1. Complete 15k run; confirm rank plateau and copy-ratio hold.
2. Phase 2: `F_e` with teacher-forced `c_plus`.
3. Phase 3: `stopgrad(c_hat)` + shuffled-c bypass test.
4. Phase 4: frame generator, multi-horizon `h_k`, inference integrator.

Minimum milestone 1 (**achieved on cerulean-snow-13**): noncollapsed `c_t`, `F_c` beats copy baseline.

---

## 11. Non-negotiable design constraints

| Constraint | Reason |
|---|---|
| Encoder frozen | Isolates hierarchy experiment; no encoder collapse |
| Target branch stop-grad + bottleneck EMA only | Stable targets; prevents self-chasing |
| `c_t` ≪ bandwidth of `e_t` | Prevents abstract state becoming a copy |
| `L_e ↛ F_c` through `c_hat` | Fine loss must not turn `c` into texture carrier |
| Frame decoder after latents; no grad into E/B | Otherwise becomes plain video generator |
| Weight `L_var` strongly enough to beat `L_flow`'s collapse pressure | **Empirical:** 0.1 insufficient; 0.5 required so far |
| Do not optimize slot diversity in isolation | Goodhart: metric up, semantics down |

---

## 12. Literature basis

| Idea | Reference |
|---|---|
| Predict latents not pixels | I-JEPA, arXiv:2301.08243 |
| Video latent prediction / frozen encoder | V-JEPA; **V-JEPA 2**, arXiv:2506.09985 |
| Rectified flow | Flow Matching, arXiv:2210.02747 |
| adaLN-Zero conditioning | DiT, Peebles & Xie 2022 |
| Variance floor (VICReg V term) | Bardes et al. — variance hinge only |
| Dataset | Something-Something V2 |

---

## 13. Empirical progress — run history (Phase 1)

Summary of meaningful runs since the original brief. Each row: what changed → what we learned → what
we changed next.

| Run | Config highlights | Finding | Fix / next |
|---|---|---|---|
| **1** | ssv2_tiny, baseline | Rank ~5/256; NaN crash at peak LR | Lower LRs, grad clip 0.5, grad-skip guard |
| **2** | Full SSv2, orthogonal queries + zero-init out_mlp | Rank ~9 then stalls; cosine ~0.72 | Added optional VICReg-C + slot loss (off by default); new diagnostics |
| **3** | `λ_slot=0.25`, k=4 | Goodhart: slot metric ok, rank worse, cosine 0.84, unstable | Drop aggressive slot; raise **k=12** |
| **4** | k=12, `λ_slot=0.05` | `L_slot` inert — loss/metric mismatch (raw vs centered cosine) | Center slots in `slot_diversity_loss` |
| **5** | k=12, centered slot loss | Slot metric ↑ but rank ↓ (~4.8), cosine 0.84, grad 10⁵ | **Drop slot loss**; `λ_var` too weak (std ~0.45) |
| **6** | **cerulean-snow-13**: k=12, **`λ_var=0.5` only** | **Win:** std→1.0, cosine→~0.20, rank rising (~13.7+), copy ratio <1, stable grads | Continue to 15k; consider gentle `λ_cov` only if rank plateaus low |

**Current best config (Phase 1 Stage 1):**

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

**Open after Run 6:** where `c_effective_rank` plateaus; whether VICReg-C is needed on top of strong
variance floor.

---

## 14. Project status (June 2026)

- **Phase:** 1, Stage 1 (coarse dynamics only).
- **Best run:** `cerulean-snow-13` — collapse reversed, beats copy baseline, training stable.
- **Not built:** `F_e`, `D`, `h_k`, inference rollout sampler.
- **Code entry points:** `train.py`, `models.py`, `losses.py`, `diagnostics.py`, `data.py`, `config.py`.
- **Prior briefs:** PDF → v0.1 → v0.2 → **this file (v0.3)**.

---

*Last updated: June 2026.*
