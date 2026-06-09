# FROZEN_ENCODER_RESEARCH.md — Choosing the frozen pretrained ViT encoder

> **What this file is.** The research and decision record for the supervisor's instruction to
> *"use a frozen pretrained ViT encoder ... from some sort of world model architecture like DINO or
> V-JEPA or similar."* It surveys the realistic candidates with hard numbers, lays out every
> decision axis, recommends **two** options, and spells out the architectural cascade the choice
> forces on the rest of the brief.
>
> **✅ DECISION LOCKED (supersedes the §8 leanings below).** Final: **encoder = V-JEPA 2 ViT-L/16**
> (`facebook/vjepa2-vitl-fpc64-256`, `D_e = 1024`); **resolution = 256×256**; **context = 8 frames**
> (→ 4 temporal tokens, `N_ctx = 1024`); **tubelet dropout = removed**.
> The human first picked ViT-B (`D_e=768`) for lean iteration, but at implementation time the ViT-B
> 2.1 checkpoint turned out to have **no `transformers`/HF repo** — it is reachable only via
> `torch.hub` (`vjepa2_1_vit_base_384`, native 384, non-`transformers` API). The HF backbones are
> ViT-L/H/g only. We therefore moved to **ViT-L via the clean HF path**, which is *better fit anyway*:
> its native resolution **256 matches our chosen resolution exactly** (no off-native RoPE stretch),
> it has first-class `get_vision_features` support, MIT license, and SOTA SSv2 features. The only cost
> is `D_e=1024` vs 768 (wider bottleneck input + `F_e` memory) and ~300M frozen params — negligible
> on an A100 since the encoder is forward-only under `no_grad`. The §6/§8 text below still reads as a
> pre-decision survey (it leans 128×128); treat those leanings as **historical** — the locked numbers
> above and in `UNDERSTANDING.md` §2.6 win.
>
> **Sources.** Meta AI V-JEPA 2 blog & arXiv:2506.09985; V-JEPA OpenReview (ICLR 2024); HF
> `transformers` `vjepa2` model docs & `facebook/vjepa2-*` model cards; DINOv2 model card &
> `facebookresearch/dinov2`; "Vision Transformers Need Registers" (arXiv:2309.16588). Retrieved
> 2026-06.

---

## 1. The requirement, restated

Replace the from-scratch `VideoViT-Small` (used today for **both** the online encoder `E` and the
EMA target encoder `E_bar`) with a **single frozen pretrained ViT**, shared by both branches. Only
the bottleneck `B` stays trainable; only `B` gets an EMA copy. Constraints from the supervisor:

1. **Pretrained**, not from scratch.
2. **Frozen** (`requires_grad = False`, always `eval()`).
3. From a **self-supervised / world-model lineage** ("DINO or V-JEPA or similar"), i.e. features
   tuned for prediction & physical understanding, not classification labels.

Our dataset is **Something-Something V2 (SSv2)** — motion- and interaction-heavy, temporal-order
sensitive — which strongly biases the choice toward **video-native** encoders.

---

## 2. Candidate survey (hard numbers)

### 2.1 Spec table

| Model | Type | Patch / tubelet | Embed dim | Layers / heads | Params | Pretrain res | Pos-embed | License | SSv2 relevance |
|---|---|---|---|---|---|---|---|---|---|
| **V-JEPA 2 ViT-L/16** | Video (JEPA) | 16 × 16, tubelet **2** | **1024** | 24 / 16 | 300M | 256 (also 384 in 2.1) | **3D-RoPE** | **MIT** | **SOTA on SSv2** |
| **V-JEPA 2 ViT-B/16** (v2.1) | Video (JEPA) | 16 × 16, tubelet **2** | **768** | 12 / 12 | 80M | 384 | 3D-RoPE | **MIT** | Strong (distilled from ViT-G) |
| V-JEPA 2 ViT-H/16 | Video (JEPA) | 16 × 16, tubelet 2 | 1280 | 32 / 16 | 600M | 256 | 3D-RoPE | MIT | SOTA-class |
| V-JEPA 2 ViT-g/16 | Video (JEPA) | 16 × 16, tubelet 2 | 1408 | 40 / — | 1B | 256/384 | 3D-RoPE | MIT | SOTA-class |
| V-JEPA (original) ViT-L/16 | Video (JEPA) | 16 × 16, tubelet 2 | 1024 | 24 / 16 | 300M | 224 | abs. 3D sin-cos | **CC-BY-NC** | Strong (original SSv2 results) |
| V-JEPA (original) ViT-H/16 | Video (JEPA) | 16 × 16, tubelet 2 | 1280 | 32 / 16 | 630M | 224 / 384 | abs. 3D sin-cos | CC-BY-NC | Strong |
| DINOv2 ViT-B/14 (+reg) | **Image** | 14 × 14 (no tubelet) | 768 | 12 / 12 | 86M | 518 (variable) | learned/interp. | **Apache-2.0** | Indirect (per-frame only) |
| DINOv2 ViT-L/14 (+reg) | Image | 14 × 14 | 1024 | 24 / 16 | 300M | 518 | learned/interp. | Apache-2.0 | Indirect (per-frame only) |

### 2.2 Also-rans (surveyed, not recommended)

- **VideoMAE / VideoMAE V2** — strong video masked-autoencoders, but they are reconstruction-trained
  (pixel targets), not JEPA; outside the supervisor's named lineage and less aligned with our
  predictive-latent thesis. Patch 16, tubelet 2, dim 768 (B). Viable fallback, not preferred.
- **Hiera** — hierarchical ViT (MAE-trained), excellent throughput, but multi-scale token geometry
  complicates a fixed `e_t` token contract. Not preferred for v0 simplicity.
- **I-JEPA** — image JEPA; same per-frame limitation as DINOv2 and weaker dense features. DINOv2 is
  the better image option if we go image-per-frame.
- **InternVideo / VideoPrism** — strong but heavier / access-gated; revisit only if V-JEPA 2 proves
  insufficient.

---

## 3. The decision axes (what actually matters)

### 3.1 Video-native vs image-per-frame — the dominant axis

- **Video-native (V-JEPA / V-JEPA 2):** ingests the whole clip and produces **spatio-temporal**
  tokens (each token spans 2 frames × 16 × 16 px). Motion is encoded *inside* the features. This is
  exactly the signal SSv2 depends on. **Strongly preferred.**
- **Image-per-frame (DINOv2):** has no notion of time. You would encode each of the 4 frames
  independently and concatenate/stack the per-frame token sets, leaving *all* temporal reasoning to
  our small trainable bottleneck. For a motion dataset this throws away the encoder's biggest
  potential advantage. Only choose if a video encoder proves impractical.

> **Verdict:** the named "world-model" lineage *and* our dataset both point to **V-JEPA 2**.

### 3.2 Patch size & how it maps to 128×128

- **V-JEPA family: patch 16** — identical to our current design. 128/16 = 8 → an 8×8 spatial grid,
  clean. **No friction.**
- **DINOv2: patch 14** — 128 is not a multiple of 14; the model crops to 126 (9×14) → a 9×9 grid.
  Slightly awkward and off-grid from the rest of our pipeline. Minor but real friction.

### 3.3 Token geometry (this sets `N_ctx` and feeds the cascade)

For a **4-frame** context with V-JEPA (tubelet 2 → 2 temporal slots):

| Resolution | Spatial grid | Temporal slots | `N_ctx` tokens | Token dim |
|---|---|---|---|---|
| 128×128 | 8 × 8 = 64 | 2 | **128** | 768 (B) / 1024 (L) |
| 224×224 | 14 × 14 = 196 | 2 | **392** | " |
| 256×256 | 16 × 16 = 256 | 2 | **512** | " |
| 384×384 | 24 × 24 = 576 | 2 | **1152** | " |

Note this is *different* from the current brief's `N_ctx = 256` (which assumed tubelet temporal = 1
over 4 frames). The tubelet-of-2 **halves** the temporal token count. This is a constant that *must*
be updated in `UNDERSTANDING.md` §2 once resolution is chosen.

For DINOv2 per-frame at 126: 9×9 = 81 patch tokens/frame × 4 frames = 324 tokens (+ CLS/registers).

### 3.4 Position-embedding type — matters for off-native resolution & short clips

- **V-JEPA 2: 3D-RoPE.** Rotary embeddings are *relative* and generalize gracefully to resolutions
  and sequence lengths the model never saw at pretraining. This is the single best property for us,
  because we want to run at **128×128** and with a **short 4-frame clip**, both off the pretraining
  regime (256/384, 64 frames). RoPE makes that far less risky.
- **V-JEPA original & DINOv2: absolute/learned pos-embeds.** Require explicit **interpolation** to a
  new resolution; more brittle off-native. Workable, but a strike against them for our 128×128 plan.

### 3.5 Embedding dim → downstream cascade cost

Every candidate is wider than our current `D_e = 384`:
- **768** (ViT-B / DINOv2-B) — moderate cascade.
- **1024** (ViT-L) — larger cascade (bottleneck input projection, fine-flow memory width).

Wider `e_t` = richer features but a bigger bottleneck input and more memory in `F_e`'s
cross-attention. See §5 for exactly what changes.

### 3.6 Short-clip / frame-count caveat

V-JEPA 2 checkpoints are `fpc64` (pretrained on **64 frames**). Feeding only **4 frames** is far
below that. RoPE tolerates it, but features may be slightly out-of-distribution. Two mitigations:
(a) accept it (cheapest), or (b) **increase context frames** (e.g. 8 or 16) to sit closer to the
pretraining regime — at the cost of more tokens and a change to `T_ctx`. This is a tunable, flagged
in §5/§8.

### 3.7 License

- **V-JEPA 2: MIT** — permissive, commercial OK. Best.
- **DINOv2: Apache-2.0** — permissive, commercial OK.
- **V-JEPA original: CC-BY-NC** — **non-commercial**. Fine for research; a liability if this ever
  needs a commercial path. A reason to prefer V-JEPA **2** over the original.

### 3.8 Tooling / availability

- **V-JEPA 2:** first-class HF `transformers` (`AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`
  → `model.get_vision_features(**inputs)` or `outputs.last_hidden_state`, shape
  `(B, seq_len, hidden)`), **and** `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')`.
  Cleanest integration.
- **DINOv2:** HF `transformers` + `torch.hub`. Clean, but image API (we'd loop frames).
- **V-JEPA original:** repo + checkpoints; less polished HF story than V-JEPA 2.

---

## 4. Resolution & compute on an A100 (answering the direct question)

**Your question:** *"With an A100, do you think matching the encoder's native resolution will add a
lot of time? Or is compute a bottleneck? If not, do you think option A (keep 128×128) is better?"*

**Short answer:** No, matching native resolution does **not** add a lot of wall-clock time on an
A100 for our setup, and the encoder is **not** the bottleneck either way — *because the encoder is
frozen*. But "keep 128×128" is still a perfectly defensible v0 choice for simplicity. Here's the
reasoning.

### 4.1 Why the frozen encoder is cheap regardless of resolution

- **Forward-only.** A frozen encoder runs under `torch.no_grad()`: no backward pass, no gradients,
  no optimizer state, minimal activation memory. Its cost is roughly **1/3** of what the same
  network would cost if it were training.
- **Tiny token counts.** With a 4-frame clip and tubelet-2, `N_ctx` is only **128 tokens @128**,
  **512 @256**, **1152 @384** (§3.3). These are *small* sequences for a ViT-L on an A100 — a forward
  pass is on the order of single-digit to low-tens of milliseconds.
- **The real bottlenecks are elsewhere:** (a) **video decoding** (decord/torchcodec on CPU workers),
  and (b) the **trainable** networks (bottleneck + flow predictor) which do forward *and* backward.
  The frozen encoder forward is unlikely to dominate at any of these resolutions.

### 4.2 The cost scaling, concretely

Self-attention is `O(N²·d)`, MLPs `O(N·d²)`. Going 128→256 multiplies tokens ~4× (128→512), so the
attention term grows ~16× and the MLP term ~4×. That sounds large, but it is 16× of a *small*
number. Blended, expect the encoder forward to be **single-digit× slower at 256 vs 128**, and that
forward is a minority of step time. 384 (~9× tokens vs 128) is more noticeable but still tractable
for a frozen forward on an A100 80GB.

### 4.3 The decisive lever: feature caching

Because `E` is **frozen**, you can **precompute `e_t` (and the target `e_{t+k}`) once for the whole
dataset and cache them to disk.** Then training reads cached features and the encoder cost during the
150k-step run is **zero** — resolution only affects a one-time precompute and storage. This makes
"match native resolution" essentially free at training time.
- **Caveat:** caching freezes the input transform. Random crop + color jitter change features
  per-epoch, so caching means either (a) using deterministic eval-style transforms, or (b) caching a
  few augmented views per clip. For SSv2-tiny this is very cheap; for full SSv2 it is a storage
  trade. This is a Phase-1 implementation choice, not a blocker.

### 4.4 Recommendation on resolution

- If we will **cache features** → **match the encoder's native resolution (256 for V-JEPA 2)** for
  best feature quality; the cost is a one-time precompute.
- If we **encode on-the-fly with augmentations** (no cache) → **128×128 is the pragmatic v0 choice**:
  cheap, simple, and V-JEPA 2's **3D-RoPE tolerates it well**. The feature-quality hit at 128 is
  modest for V-JEPA 2 specifically (much worse for absolute-pos-embed models).
- **Quality ranking is not the point of v0.** v0 exists to validate the *hierarchy mechanism*. Given
  that, **Option A (128×128) is a reasonable default** — *provided we pick V-JEPA 2 (RoPE)*. If you
  want the cleaner "use the features as pretrained" story and we add caching, go 256.

> My lean: **V-JEPA 2 at 128×128 on-the-fly for the first v0 run** (cheapest path to a hierarchy
> signal), with **256 + feature caching** as the documented quality upgrade. Final call is yours
> (§8).

---

## 5. The architectural cascade (what each choice forces to change)

Whatever we pick, swapping the encoder forces these edits in `UNDERSTANDING.md` / `BRIEF_V0_2.md` /
`PHASE_1.md`:

| Area | Current (`v0.1`) | After frozen-encoder swap |
|---|---|---|
| `E` / `E_bar` | Two trained VideoViT-S (dim 384) | **One frozen** pretrained ViT (dim 768 or 1024), shared |
| `D_e` (detailed dim) | 384 | **768** (ViT-B) or **1024** (ViT-L) — set by the pick |
| `N_ctx` (context tokens) | 256 (tubelet T=1) | **128 / 512 / 1152** per resolution (tubelet T=2) |
| Target representation | encode single frame `y` (64 tokens) | encode clip `x_{≤t+k}` → `e_{t+k}` (same geometry as context) |
| EMA scope | `E_bar` + `B_bar` | **`B_EMA` only** (encoder frozen, shared) |
| Encoder LR | 2e-4 from scratch | **none** (frozen) |
| Tubelet dropout | 40% on context | **OPEN** — a frozen pretrained encoder did not see dropped tokens at pretraining; likely **drop or reduce** it, or move masking into the bottleneck instead (decision item) |
| Bottleneck `B` input proj | 384 → 256 | `D_e` → 256 (768→256 or 1024→256) |
| Fine flow `F_e` memory | `concat(e_t[384], proj_c)` | `concat(e_t[D_e], proj_c)`; cross-attn keys/values widen |
| Collapse loss | SIGReg(e_t) + SIGReg(c_t) | **variance floor on c_t only** (SIGReg removed) |
| Pos-embeds | our own 3D sin-cos | **the encoder's own** (RoPE for V-JEPA 2) — we stop adding our own to `e_t` |
| Pixel normalization | [−1, 1] | **the encoder's expected normalization** (V-JEPA processor / ImageNet stats) — must match the pretrained model, *separate* from the VAE's [−1,1] for Stage 4 |

Two of these deserve a human decision and are listed in §8: **tubelet dropout under a frozen
encoder**, and **context frame count** (§3.6).

---

## 6. Recommendation: two options

### Option A (primary) — **V-JEPA 2 ViT-L/16**

- **Why:** video-native, JEPA lineage (our exact family), patch 16 (matches us), **3D-RoPE**
  (handles 128×128 and short clips best), **MIT** license, first-class HF support, and **SOTA on
  SSv2 — our dataset.** dim 1024 gives the richest `e_t`.
- **Cost:** `D_e = 1024` → the largest downstream cascade (widest bottleneck input and `F_e` memory),
  ~300M frozen params. Comfortable on A100 80GB (frozen, forward-only).
- **Best when:** we want the strongest features and don't mind the wider cascade.

### Option B (lean / fast-iteration) — **V-JEPA 2 ViT-B/16** (V-JEPA 2.1, 80M)

- **Why:** same family and all of A's advantages (RoPE, patch 16, MIT, HF), but **dim 768** and only
  80M params → smaller cascade, faster forward, less memory, quicker debug cycles for v0. It is a
  *distilled* model (from ViT-G), so quality is strong for its size.
- **Cost:** slightly weaker features than ViT-L; native resolution is 384 (so 128 is further
  off-native, though RoPE mitigates).
- **Best when:** we prioritize iteration speed and a lean v0.

### Image-only fallback — **DINOv2 ViT-B/14 (+registers)**

- Only if a video encoder proves impractical. Apache-2.0, excellent dense features, but **no
  temporal modeling** (per-frame), **patch 14** (off our 16-grid), and it pushes *all* motion
  reasoning onto the bottleneck. Not recommended for a motion dataset, but documented as the escape
  hatch.

---

## 7. Quick integration sketch (V-JEPA 2 via HF)

```python
# Reference only — not committed code (Phase 1 will implement properly).
from transformers import AutoModel, AutoVideoProcessor

REPO = "facebook/vjepa2-vitl-fpc64-256"          # or vjepa2-vitb-* for Option B
encoder   = AutoModel.from_pretrained(REPO, attn_implementation="sdpa").eval()
processor = AutoVideoProcessor.from_pretrained(REPO)
for p in encoder.parameters():
    p.requires_grad = False                       # FROZEN

# context_clip: (B, T, C, H, W) sampled per our data pipeline
inputs = processor(context_clip, return_tensors="pt")
with torch.no_grad():
    e_t = encoder.get_vision_features(**inputs)    # (B, N_ctx, D_e=1024)
```

Notes: the processor enforces the model's expected normalization/resolution; tubelet-2 requires an
even frame count; `last_hidden_state` is equivalent to `get_vision_features`.

---

## 8. Open decisions for the human (follow-up)

These cannot be resolved by research alone — they are your calls, and they lock the numeric rewrites:

1. **Encoder pick:** Option A (V-JEPA 2 ViT-L/16, dim 1024) or Option B (V-JEPA 2 ViT-B/16, dim 768)?
   *(My lean: start with B for fast iteration, then validate on A — but A if you want best features
   from the start.)*
2. **Resolution:** 128×128 on-the-fly (cheap v0) or 256 with feature caching (best features)?
   *(My lean: 128×128 on-the-fly for the first run, given V-JEPA 2's RoPE.)*
3. **Tubelet dropout under a frozen encoder:** keep 40% / reduce / drop entirely / move masking into
   the bottleneck? *(My lean: drop tubelet dropout on the frozen encoder input; if we want a
   shortcut-prevention knob, apply light masking inside the trainable bottleneck instead.)*
4. **Context frame count:** keep 4 frames, or increase (e.g. 8/16) to sit closer to V-JEPA 2's
   pretraining regime? *(My lean: keep 4 for v0 simplicity; revisit if features look degraded.)*

Answer these and I lock `UNDERSTANDING.md`, `PHASE_1.md`, and `BRIEF_V0_2.md` to the chosen numbers.
