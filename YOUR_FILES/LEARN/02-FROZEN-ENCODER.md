# 02 — The Frozen Encoder: ViTs, V-JEPA 2, Tubelets, and Why We Don't Train It

> **What you'll understand after this file:** exactly how 8 frames of pixels
> become 1024 tokens of 1024 dims, what a tubelet is, what V-JEPA 2 was
> trained to do, and the full argument for freezing it (it's the single most
> consequential decision in v0.2).

---

## 1. Vision Transformers in 90 seconds (the parts we need)

A **Vision Transformer (ViT)** treats an image the way a language model
treats a sentence: chop it into pieces ("patches"), embed each piece as a
vector ("token"), and run transformer blocks (self-attention + MLP) over
the token sequence. "ViT-L/16" decodes as:

- **L** = Large: 24 transformer blocks, embedding dim 1024, ~300M params.
- **/16** = each patch is 16×16 pixels.

For a 256×256 image: 256/16 = **16 patches per side** → a 16×16 grid →
**256 tokens per frame**. Each token is a 1024-dim vector after the
embedding projection and transformer blocks. Self-attention lets every
token look at every other token, so by the final layer each token encodes
its patch *in the context of the whole image*.

## 2. From images to video: tubelets

Video adds time. The naive extension — tokenize every frame separately —
wastes tokens, because adjacent frames are nearly identical. Video ViTs
instead use **tubelets**: small 3D blocks of `(time × height × width)`
pixels. V-JEPA 2 uses tubelets of **2 frames × 16 × 16 pixels** — pairs of
co-located patches from consecutive frames merged into a single token.

Run the geometry for our clips (8 frames, 256×256):

```
temporal:  8 frames / tubelet-2          = 4 temporal slots
spatial:   (256/16) × (256/16)           = 16 × 16 = 256 tokens per slot
total:     4 × 256                       = 1024 tokens, each 1024-dim
```

This is where the magic numbers in `config.py` come from — they're not
choices, they're *derived* from the encoder's fixed geometry:

```72:95:config.py
    @property
    def grid_spatial(self) -> int:
        """Patch-grid side length per frame, H / encoder_patch (256/16 = 16)."""
        return self.h // self.encoder_patch

    @property
    def n_temporal_tokens(self) -> int:
        """Temporal tokens after tubelet merging, T / tubelet (8/2 = 4)."""
        return self.t_ctx // self.encoder_tubelet

    @property
    def tokens_per_frame(self) -> int:
        """Spatial tokens per temporal slot, grid_spatial**2 (16*16 = 256)."""
        return self.grid_spatial * self.grid_spatial

    @property
    def n_ctx(self) -> int:
        """Context token count from encoder geometry, (T/2)*(H/16)**2 = 1024."""
        return self.n_temporal_tokens * self.tokens_per_frame
```

**Token ordering matters.** The encoder emits tokens *temporal-major*:
index = `t·(16·16) + h·16 + w`. Token 0 is the top-left of temporal slot 0;
token 256 is the top-left of temporal slot 1. The bottleneck depends on
this ordering to reassemble tokens into spatial grids (file 03). If this
assumption were wrong, training would still "work" — the loss would go
down — but the ConvNeXt mixing would be convolving over scrambled
neighborhoods. This is the kind of silent bug the project's contract tests
exist to prevent.

A note on why a *single* token can carry motion: a tubelet token sees two
consecutive frames, so frame-to-frame differences (= instantaneous motion)
are visible *within one token*. Longer-range motion across the clip is
assembled by self-attention across the 4 temporal slots.

## 3. What V-JEPA 2 is and what it learned

**V-JEPA 2** (Meta, 2025) is a video encoder pretrained with the JEPA
recipe from file 01, at scale: mask out chunks of a video, encode the
visible part, and predict the *latents* of the masked part (targets come
from an EMA copy of the encoder — the same trick we use, see file 05). It
was trained on millions of internet videos. We use the checkpoint
`facebook/vjepa2-vitl-fpc64-256`:

- `vitl` — ViT-Large, embedding dim **D_e = 1024**.
- `fpc64` — pretrained with 64 frames per clip (it generalizes to our 8).
- `256` — native resolution 256×256, exactly ours, so no resizing mismatch.

Because its pretraining objective was *prediction in latent space*, its
features are biased toward exactly what we want: motion, object
permanence, spatiotemporal structure — not just static appearance. This is
why it's a better donor than, say, an image CLIP applied frame-by-frame.

The wrapper in our code is deliberately thin:

```77:87:models.py
    @torch.no_grad()
    def forward(self, clip: Tensor) -> Tensor:
        """Encode a pixel clip into detailed tokens (no grad).

        Args:
            clip: (B, T, C, H, W) encoder-normalized pixel clip.
        Returns:
            detailed: (B, N_ctx, D_e) per-tubelet features.
        """
        features = self.model.get_vision_features(clip)
        return features
```

We do **not** implement patching, position embeddings, or tubelet
projection — the pretrained model owns all of it internally (including its
3D rotary position encoding). Our contract with it is purely shape-level:
`(B, 8, 3, 256, 256)` normalized pixels in, `(B, 1024, 1024)` tokens out.

**Normalization is part of the contract.** The encoder expects ImageNet
statistics — `mean (0.485, 0.456, 0.406)`, `std (0.229, 0.224, 0.225)` —
not `[-1, 1]`. Feed it `[-1, 1]` and you get no error, no NaN, just
quietly degraded features. The dataloader (`data.py::_normalize_encoder`)
applies the correct stats, and the smoke test asserts the output range is
*not* `[-1, 1]` to catch regressions. Lesson to generalize: **pretrained
models have invisible input contracts, and violating them fails silently.**

## 4. Why frozen? The complete argument

In v0.1, the plan was to train our own encoder from scratch. The
supervisor's v0.2 directive replaced it with a frozen pretrained one. This
is the most consequential design decision in the project. The reasoning:

**(a) It deletes the hardest failure mode.** A *trainable* encoder in a
JEPA can collapse: it controls both the input representation and
(through the EMA copy) the target, so "map everything to the same vector"
is a global optimum of the prediction loss. Preventing that requires
machinery — target encoders, masking strategies, covariance regularizers
(VICReg/SIGReg), careful EMA schedules — each with its own
hyperparameters and its own ways to go wrong. A *frozen* encoder cannot
collapse, cannot drift, cannot cheat. The line in `diagnostics.py` —
*"The encoder is frozen, so `e_t` cannot collapse and is not monitored"* —
is an entire monitoring subsystem that no longer needs to exist.

**(b) It isolates the research question.** Phase 1's question is "does a
flow-matched bottleneck latent learn coarse dynamics?" With a trainable
encoder, a failure is ambiguous: bad encoder? bad bottleneck? bad
predictor? With a frozen encoder, `e_t` is a *fixed, known-good* input
distribution, and any failure is attributable to the parts we built.
Scientific control, applied to architecture.

**(c) It's enormously cheaper.** ~300M encoder params need no gradients,
no optimizer state (AdamW keeps 2 extra tensors per param — that's ~2.4GB
of GPU memory saved at fp32 state), and no backward pass through 24
transformer blocks. Trainable params drop from ~hundreds of millions to
~15M (bottleneck + flow). This is what makes a single-GPU RunPod budget
viable.

**(d) Someone already spent the compute.** Meta trained V-JEPA 2 on more
video than we will ever process. Re-learning generic visual features from
~170k SSv2 clips would produce strictly worse features at vastly greater
cost.

**The cost we accept:** `e_t` is not adapted to SSv2 or to our objective.
If V-JEPA 2's features lack something our task needs, we cannot recover it
— the bottleneck can only select and recombine what's there, never add to
it. This is a known, accepted ceiling. (Research on frozen-encoder
pipelines — see `AGENT_FILES/KNOWLEDGE/FROZEN_ENCODER_RESEARCH.md` —
supports that V-JEPA-class features are rich enough for dynamics tasks.)

## 5. The engineering of staying frozen

"Frozen" must be *enforced*, not assumed. Three mechanisms in `models.py`:

1. **Constructor:** every param gets `requires_grad = False`, then a check
   raises if the trainable count isn't exactly 0.
2. **`train()` override:** PyTorch's `model.train()` recursively flips all
   children to train mode — which would re-enable dropout and batch-norm
   updates inside the encoder. The override forces `self.model.eval()`
   back, every time.
3. **`@torch.no_grad()` on forward:** even with frozen params, autograd
   would otherwise *record* the encoder forward for the backward graph,
   wasting memory. `no_grad` skips graph construction entirely.

And one mechanism in `train.py::run_stage0`: before any real training, it
snapshots an encoder parameter, runs a full train step, and asserts the
parameter is bit-identical after. Belt, suspenders, and a test for the
belt. The general MLOps lesson: **invariants you rely on should be
asserted at runtime, not trusted.**

One nice consequence: checkpoints don't contain the encoder at all
(`save_checkpoint` saves only bottleneck, EMA bottleneck, flow, optimizer)
— it's always re-loaded from the HuggingFace cache. Checkpoints stay
~hundreds of MB instead of ~1.5GB.

## 6. What `e_t` actually looks like

Shape `(B, 1024, 1024)`: batch × tokens × features. Conceptually a
4×16×16 spatiotemporal grid of descriptors, each saying "here's what's at
this place and moment, in context." It is:

- **deterministic** given the clip (encoder is in eval mode, no dropout);
- **identical for both branches** — the same frozen instance encodes the
  context clip (→ `e_t`) and the future clip (→ `e_plus`); only the
  *bottleneck* differs between branches (online vs EMA);
- **never stored** — recomputed every step. (A tempting optimization is to
  precompute and cache all `e_t` to disk since the encoder never changes;
  the trade-off is that augmentations — random crop, color jitter — happen
  in pixel space *before* encoding, so caching would freeze augmentations
  too. This is a real design tension; see file 08.)

## 7. Questions to test yourself

1. Why does an 8-frame clip produce 1024 tokens and not 2048? *(Tubelet-2
   halves the temporal dimension: 4 slots × 256 spatial = 1024.)*
2. If you switched to 16-frame clips, what breaks? *(n_ctx becomes 2048;
   the bottleneck's grid reshape and the n_ctx assertion must change;
   encoder handles it fine since it was pretrained on 64 frames.)*
3. Why is there no `lr_encoder` in `TrainConfig`? *(Frozen params are
   never given to the optimizer — see `make_optimizer`, which only has
   param groups for B and F_c.)*
4. What would happen if `train()` weren't overridden? *(`.train()` calls
   from the outer loop would put the encoder in train mode; for this ViT
   the practical risk is dropout becoming active, making `e_t` stochastic
   and the EMA target noisy.)*
