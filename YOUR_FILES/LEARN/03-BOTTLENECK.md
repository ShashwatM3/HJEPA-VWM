# 03 — The Bottleneck: How 1024 Tokens Become 32, and Why Every Sub-Module Exists

> **What you'll understand after this file:** the exact dataflow inside the
> bottleneck `B`, what ConvNeXt blocks and cross-attention queries are, why
> *this* combination was chosen, what initialization policy governs it, and
> the open question about its effective rank.

---

## 1. The job description

Input: `e_t`, shape `(B, 1024, 1024)` — 1024 detailed tokens.
Output: `c_t`, shape `(B, 32, 256)` — 32 abstract tokens.

That's a 128× reduction in numbers (1,048,576 → 8,192 per clip). The
bottleneck must decide *what survives compression*. Crucially, it is not
told what to keep — it is trained only through two pressures:

1. **L_flow** (backpropagated from the predictor): "produce a `c_t` that
   makes the future predictable." Useful information about dynamics →
   keep. Static noise → drop.
2. **L_var** (the variance floor): "don't make `c_t` constant."

Everything inside the module is engineering to give those pressures a
flexible, well-conditioned function to shape.

> Two *optional* regularizers also exist now (Phase 04): `covariance_floor`
> (VICReg-C, decorrelate feature dims) and `slot_diversity_loss` (push the 32
> slots apart). Both default **off** (`lambda_cov = lambda_slot = 0.0`) and
> are being trialed empirically against the dimensional-collapse problem —
> see §7. When off, the two pressures above are the only ones.

## 2. The dataflow, line by line

```171:209:models.py
    def forward(self, detailed: Tensor, *, return_attn: bool = False):
        """Compress detailed tokens to abstract tokens.
        # ... docstring: return_attn=True also returns per-head attn weights
        #     (B, num_heads, N_c, N_ctx) for diagnostics; training leaves it False ...
        """
        b, n, _ = detailed.shape
        cfg = self.cfg
        if n != cfg.n_ctx:
            raise ValueError(f"Expected {cfg.n_ctx} tokens, got {n}")
        mix = cfg.bottleneck_mixer_dim
        tokens = self.in_proj(detailed)
        t, g = cfg.n_temporal_tokens, cfg.grid_spatial
        # Temporal-major token order from the encoder: index = t*(g*g) + h*g + w.
        grid = tokens.reshape(b * t, g, g, mix).permute(0, 3, 1, 2)
        mixed = self.mixers(grid).permute(0, 2, 3, 1).reshape(b, t * g * g, mix)
        memory = self.to_kv(mixed)
        queries = self.queries[None].expand(b, -1, -1)
        attended, attn = self.cross_attn(
            queries, memory, memory,
            need_weights=return_attn, average_attn_weights=False,
        )
        abstract = self.norm(attended + self.out_mlp(attended))
        if return_attn:
            return abstract, attn
        return abstract
```

Five stages. Let's take each one and answer *what / why / what-if-removed*.

### Stage 1 — Input projection: `in_proj` (Linear 1024 → 256)

**What:** a per-token linear map from the encoder's width to the
bottleneck's working width (`bottleneck_mixer_dim = 256 = d_c`).

**Why:** (a) the frozen encoder's 1024 dims are sized for *its* job, not
ours — working at 256 cuts all downstream compute 4×; (b) it's the first
trainable layer touching frozen features, so it learns *which linear
combinations of V-JEPA features matter for our objective*. Think of it as
a learned feature-selection layer.

**If removed:** everything downstream is 4× wider and slower, and you lose
the cheap re-weighting of frozen features.

### Stage 2 — Grid reshape: tokens → spatial maps

**What:** `(B, 1024, 256)` → `(B·4, 256, 16, 16)`. The 1024-token sequence
is split into its 4 temporal slots, and each slot's 256 tokens are laid
back out as the 16×16 spatial grid they came from (channels-first for
conv layers).

**Why:** the next stage is convolutional, and convolution only makes sense
on data with spatial layout. This reshape is valid *only because* the
encoder emits tokens temporal-major (`index = t·256 + h·16 + w`) — the
single most load-bearing assumption in the module, which is why the code
comments it explicitly and `tests/test_phase1_contract.py` pins it.

Note `B·4`, not `B`: each temporal slot is mixed *independently*, with
**shared weights** across slots (the same ConvNeXt blocks process slot 0
and slot 3). Spatial mixing is per-moment; *temporal* combination is
deferred to cross-attention in stage 4.

### Stage 3 — ConvNeXt blocks (×2): local spatial mixing

**What is ConvNeXt?** A 2022 ConvNet design ("A ConvNet for the 2020s")
that borrows transformer ideas. One block:

```
input ──► depthwise 7×7 conv ──► LayerNorm ──► Linear(×4) ──► GELU ──► Linear(÷4) ──► + input
```

- **Depthwise conv** (`groups=dim`): each of the 256 channels is convolved
  with its *own* 7×7 filter — spatial mixing without channel mixing. Cheap
  (256·49 weights vs 256²·49 for a full conv) and gives each token a 7×7
  neighborhood view.
- **Pointwise MLP** (the two Linears): channel mixing without spatial
  mixing, with a 4× hidden expansion — exactly a transformer FFN.
- **Residual** (`+ input`): the block computes a *correction* to identity,
  which keeps gradients well-behaved and makes "do nothing" easy to learn.

So one block = "look at your 7×7 neighborhood, then recombine your
channels." Two blocks stack the receptive field to ~13×13 of the 16×16
grid — nearly global — for a tiny parameter cost.

**Why have this stage at all?** The docstring says it: *"Local spatial
context before query compression helps `c_t` retain future-relevant
structure rather than raw per-token noise."* The intuition: cross-attention
(next stage) computes a **weighted average** of tokens. Averaging raw
per-patch features blurs precisely the local relationships (edges between
moving object and background, contact points) that predict motion. The
ConvNeXt stage lets tokens first *absorb their neighborhood* so that what
gets averaged is already locally contextualized. Conv layers also bring a
locality **inductive bias** — nearby patches relate more than distant ones
— which is true of video and free to exploit.

**If removed:** the model still runs; attention would have to learn
locality from scratch through attention weights, which is slower and
weaker with only ~15M trainable params and a small dataset. (This is a
testable claim — an ablation candidate.)

**Why not 3D conv over time too?** Only 4 temporal slots exist, and the
tubelets already encode 2-frame motion internally; cross-attention handles
the rest. Cheaper and simpler wins.

### Stage 4 — Cross-attention from 32 learned queries

**What:** 32 learned vectors (`self.queries`, shape `(32, 256)` — a free
`nn.Parameter`, the same for every input) attend over the 1024 mixed
tokens (after `to_kv` maps them into the 256-dim key/value space):

```
attention(Q = 32 learned queries, K = V = 1024 mixed tokens) → 32 outputs
```

Each query computes similarity against all 1024 tokens, softmaxes, and
takes a weighted average of them. Each of the 32 outputs is therefore a
*learned, input-dependent summary* of the whole clip.

**Why is this the right compression operator?** Compare the alternatives:

| Operator | What it does | Problem |
|---|---|---|
| Mean/max pooling | fixed average over tokens | not learned, not content-adaptive; one summary, not 32 specialized ones |
| Strided conv downsampling | keep a coarser grid | output stays *spatial*; you get "the top-left region" not "the moving object" |
| Take first 32 tokens | arbitrary | discards 97% of the clip by position |
| **Learned-query cross-attention** | each query learns *what kind of thing to look for*, then finds it wherever it is | — |

The learned-query pattern (a.k.a. Perceiver-style latent bottleneck,
DETR-style object queries) is the standard answer when you need
"fixed-size, content-adaptive summary of a variable/large token set."
Queries tend to *specialize* during training — one may track dominant
motion, another hand position, another scene layout. That specialization
is learned, not designed.

This is also why the answer to "why attention *again*, didn't the ViT
already do attention?" is: different job. The ViT's *self*-attention
contextualizes 1024 tokens among themselves (1024→1024). Our
*cross*-attention is asymmetric: 32 probes reading from 1024 memories
(1024→32). Same mechanism, opposite purpose — one enriches, one distills.

**If removed/replaced with pooling:** you'd lose input-adaptive selection;
the bottleneck would compress by *averaging* rather than by *choosing*,
and fine-but-critical signals (small moving hand in a large static scene)
would be diluted.

### Stage 5 — Output MLP + LayerNorm

**What:** `attended + out_mlp(attended)` — a residual 4×-expansion MLP
(LayerNorm → Linear 256→1024 → GELU → Linear 1024→256) — then a final
LayerNorm.

**Why the MLP:** attention's output is a weighted average — a fundamentally
*linear* combination of values. The MLP adds per-slot nonlinear
post-processing, letting each slot compute functions of its gathered
evidence rather than just report the average. (Exactly why every
transformer block pairs attention with an FFN.)

**Why the final LayerNorm:** it pins the *scale* of `c_t`. Downstream, the
flow predictor mixes `c_t` with Gaussian noise of std 1; the variance
floor hinges per-dim std at 1.0; the EMA target must live in the same
numeric range as the online output. A normalized output keeps all three
consistent and prevents the bottleneck from "cheating" the variance floor
by inflating scale.

## 3. Parameter budget (intuition for scale)

Rough counts: `in_proj` ≈ 0.26M; 2 ConvNeXt blocks ≈ 0.55M; `to_kv` ≈
0.07M; queries ≈ 8k; cross-attention ≈ 0.26M; out_mlp ≈ 0.53M. **Total ≈
1.7M params** — tiny next to the 300M frozen encoder and ~13M flow
predictor. The bottleneck is deliberately small: it is a *selector*, not a
*computer*. The heavy representation work was prepaid by V-JEPA 2.

## 4. Initialization policy (and why your tech lead cares)

Initialization decides where optimization *starts*; bad starts can be
unrecoverable (saturation, collapse, instability). Current policy:

| Component | Init | Rationale |
|---|---|---|
| FrozenEncoder | pretrained weights | the whole point of v0.2 |
| Bottleneck queries | **`nn.init.orthogonal_`** (unit-norm rows) | the 32 queries start mutually perpendicular *and* at unit scale, so attention logits are non-trivial and each slot reads a distinct, non-redundant summary from step 0 |
| Bottleneck `out_mlp` last layer | **zeros** (weight + bias) | adaLN-Zero-style identity start: the residual MLP begins as a pass-through, so early training isn't destabilized by random residual contributions |
| Bottleneck other Linears/convs | PyTorch defaults (Kaiming-uniform) | standard; reasonable variance preservation through GELU nets |
| Flow `AdaLNBlock.mod` final layer | **zeros** | adaLN-Zero (see file 04): each flow block starts as exact identity |

> **Phase-04 update — these init fixes are now baked in, not optional.**
> Earlier (v0.2) the queries used `randn · 0.02` and `out_mlp` used PyTorch
> defaults. Both were replaced as part of the dimensional-collapse work and
> are now hardcoded defaults in `Bottleneck.__init__` (no config flags — they
> are *fixes*, not empirical knobs). The code comments tag them "Fix 1" and
> "Fix 2".

Why the query init connects to the rank/collapse problem — and the subtle
correction worth internalizing: the old `randn · 0.02` problem was **not**
that the queries pointed the same direction (random 256-d vectors are
already nearly orthogonal). The real problem was **scale**: with norm
≈ 0.02·√256 ≈ 0.3, the attention logits `q·k` were tiny, so softmax was
near-uniform for *every* slot, and all 32 slots read out approximately the
same mean token — identical outputs despite distinct directions. `orthogonal_`
fixes this mostly because its unit-norm rows are a ~50× scale increase (the
orthogonality is a bonus, not the main lever). This sharpens the *starting*
attention, but note: init governs where you start; it does not by itself
prevent the optimizer from rediscovering a low-rank solution later — that is
a training-dynamics / objective problem (see §7).

## 5. The EMA twin

There are *two* copies of this module in memory: the online `Bottleneck`
(trained by gradients) and the `TargetBottleneck` (updated only by EMA,
produces the prediction target `c_plus`). Architecturally identical, byte
-for-byte the same code path — only the update rule differs. File 05
explains why the twin exists. For now, one consequence matters: any change
you make to the bottleneck architecture automatically applies to the
target too (it's constructed via `deepcopy`), so there is no
"target drift" risk at the architecture level.

## 6. What `c_t` is, semantically

Nobody assigns meaning to the 32 slots; meaning *emerges* from the
training pressure. The honest description: **`c_t` is whatever 8,192
numbers make the near future most predictable while staying non-constant.**
If Phase 1 works, that ends up encoding "what is where and how it's
moving." Whether it actually does is checked empirically — by the
acceptance baselines and latent-health metrics (file 06), never by
inspection or hope.

## 7. The open problem: effective rank ~5/256

In Run 1, `c_effective_rank` plateaued around ~5 (max possible 256). The
latent was *not* fully collapsed (variance alive, cross-video cosine OK)
but used ~2% of its dimensional capacity — like buying a 256-lane highway
and using 5 lanes. Three candidate explanations were on the table, each
with a different fix:

1. **Dataset too small** (~4k clips in ssv2_tiny; ~240 epochs in a 15k-step
   run): there may genuinely be only ~5 axes of variation the model needs.
   Fix: run on full SSv2. Lowest risk, tests data before architecture.
2. **Init/optimization artifact**: queries read out near-identical mean
   tokens early. Fix: orthogonal query init, zero-init out_mlp (§4).
3. **Missing regularizer**: the variance floor only prevents *constant*
   `c_t`; it never asks for *decorrelated* dims, nor for *distinct slots*.
   Fix: add a covariance term (VICReg-C) and/or a slot-diversity term.

**Where this stands now (Phase 04 in progress):**

- Fix (2) is **done** — both init fixes are baked in (§4).
- Fix (1) is **done** — a baseline was run on full SSv2. Rank rose modestly
  (to ~9 at step ~3.5k) but did not climb to the dimensionality we want, so
  "data alone" is not the whole story.
- The diagnostics were sharpened to localize the failure: `c_slot_diversity_rank`
  (within-video effective rank of the 32 slot outputs) and per-head
  `c_attn_entropy` were added. The dominant remaining symptom is **slot
  collapse + near-uniform attention** — slots and heads stay too similar /
  too diffuse — rather than only feature-dimension correlation.
- Fix (3) is now being **trialed empirically** (not as a fully-committed
  default): a *gentle* `covariance_floor` (VICReg-C, `lambda_cov`) and a
  `slot_diversity_loss` (`lambda_slot`) exist in `losses.py` and are wired
  into `train_step`; both are always logged but only added to the loss when
  their `lambda` > 0. An over-aggressive first attempt (`lambda_slot=0.25`)
  moved the slot metric but hurt prediction and destabilized training
  (Goodhart behavior), so the current direction is *gentle* weights plus a
  **harder task** — increasing the prediction horizon (`horizon_k`) so the
  task can no longer be solved by a low-information latent.

The lesson so far: init governs the *start*; the variance floor prevents
*constant* output; but neither forces the latent to *use* its capacity. A
weak (too-easy / too-overlapping) prediction task is what lets a low-rank
latent survive — which is why the horizon change matters as much as the
regularizers. Watch `c_effective_rank`, `c_slot_diversity_rank`,
`c_attn_entropy`, and the `coarse_vs_copy_ratio` *together* — the
regularizers can satisfy a single metric without improving the actual task.

## 8. Questions to test yourself

1. Why do ConvNeXt blocks run on `B·4` grids instead of `B`? *(Each
   temporal slot is mixed independently with shared weights; temporal
   combination is cross-attention's job.)*
2. What two facts make the grid reshape in stage 2 valid? *(Temporal-major
   token order from the encoder, and 256 = 16² spatial tokens per slot.)*
3. Why can't the bottleneck "cheat" the variance floor by scaling up its
   output? *(Final LayerNorm pins the per-token scale.)*
4. Your tech lead asks "can we just mean-pool e_t to get c_t?" — what do
   you lose? *(Content-adaptive selection; 32 specialized summaries
   collapse to 1 fixed average; small moving objects get diluted.)*
5. Why is reintroducing SIGReg a last resort rather than the obvious fix
   for low rank? *(Supervisor directive for v0.2 minimalism; extra
   hyperparameters; risk of optimizing the diagnostic; rank may rise on
   its own with more diverse data.)*
