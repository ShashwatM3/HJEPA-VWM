# Design plan — preserve V-JEPA information before the `32 x 256` bottleneck

This is the architecture-design artifact for the next Investigation 016 reconstruction thread. It
turns three proposed changes into one coherent, minimally invasive design:

1. remove fixed ZCA whitening;
2. widen the bottleneck's internal memory/slot stream from 256 to 512 or 1,024;
3. move the only projection to the external `D_c=256` abstract width until after the complete
   cross-attention latent processor.

This file began as the design plan. The architecture described here was implemented on 2026-07-18;
the executable two-arm bundle now has separate records for
[`M=512`](run_064_unwhitened_internal_memory_m512/) and
[`M=1024`](run_065_unwhitened_internal_memory_m1024/), with the shared launch in [`GUIDE.md`](GUIDE.md).
Results remain pending.
The empirical basis is the end-to-end
[`reconstruction_floor_architecture_audit`](reconstruction_floor_architecture_audit/ANALYSIS.md),
the completed
[`N_c=32/64/128` capacity sweep](bottleneck_slot_capacity_sweep/ANALYSIS.md), and the
[`lambda_recon=1` analysis](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md).
Code and `AGENT_FILES/AGENTS.md` remain authoritative for current behavior.

---

## Execution decision — 2026-07-18

The human selected a combined implementation rather than the longer single-delta sequence proposed
later in this document. The paid bundle therefore has exactly two EGO4D arms:

| Arm | Whitening | Complete internal width | Projection | Geometry weights |
|---|---|---:|---|---|
| 1 | off | 512 | one `512 -> 256` map after all three latent blocks | all zero |
| 2 | off | 1,024 | one `1024 -> 256` map after all three latent blocks | all zero |

Both retain `N_c=32`, external `D_c=256`, the 512-by-4 decoder, absolute cosine reconstruction,
`lambda_recon=1`, and the existing schedule. The two runs execute sequentially on the one A100.

This decision supersedes Section 8's recommendation to run an unwhitened 256 control first. It
means the experiment cleanly isolates 512 versus 1,024, but it does not independently isolate
whitening removal or late projection from historical controls. That attribution cost is explicit,
not accidental. Existing whitening artifacts remain on the volume but are never read because every
`--whiten-*` flag is omitted.

The success rule is also joint: raw and whitened cosine values are not numerically interchangeable,
and a lower correct-code loss counts as preservation only if the correct-versus-shuffled code gap
improves. See the 1,024 arm's
[`PLAN.md`](run_065_unwhitened_internal_memory_m1024/PLAN.md) for the preregistered effect/cost guard.

The previously recommended source-diverse diagnostic repair is not part of these three architecture
changes. Consequently, this launch may conclude only that exact-chunk preservation improved within
the recorded source. It is prohibited from claiming global cross-source or global-video
preservation; that stronger claim remains gated on source-unique validation and cross-source code
derangement.

---

## 0. Executive decision

The three ideas are directionally correct, but changes 2 and 3 are architecturally coupled. Before
the 2026-07-18 implementation, increasing `bottleneck_mixer_dim` alone did **not** create wide
attention memory: `Bottleneck.to_kv` immediately mapped that wider mixer output back to `D_c=256`,
and the queries and all three latent blocks still operated at 256. A run on that pre-change code
would have tested wider local ConvNeXt preprocessing, not the proposed wide information path.

The smallest coherent implementation is therefore:

```text
external interface (unchanged)
    detailed e: (B, 1024, 1024)
        -> Bottleneck internals at M in {256, 512, 1024}
        -> final learned projection M -> D_c=256
    abstract c: (B, 32, 256)
```

Use the existing `ModelConfig.bottleneck_mixer_dim` as the single **internal width** `M` throughout
the input projection, ConvNeXt memory, `to_kv`, learned queries, and latent processor. Add a final
projection only when `M != D_c`, immediately before the existing final `LayerNorm(D_c)`. At the
shipped `M=D_c=256`, make that projection an identity; the default forward computation and
historical state dictionary remain unchanged.

The scientific and engineering recommendations differ slightly:

- **512 is the recommended first practical width.** It halves the initial channel-compression
  ratio, retains twice as many channels, makes the whole attention processor twice as wide, and
  keeps cost within a plausible first paid run.
- **1,024 is the clean scientific upper-bound arm.** It is the only choice that literally avoids
  any dimension-reducing channel projection before global pooling, but it makes the bottleneck's
  quadratic parameter blocks about sixteen times their 256-D cost.
- If only one **width** arm can be afforded, choose 512 for engineering progress. If the purpose is
  to settle whether *any* early channel squeeze causes the floor, 1,024 is the decisive width arm.

The original attribution-first recommendation was to isolate whitening before widening. The
2026-07-18 execution decision above deliberately combines those changes, so only the 512-to-1,024
width comparison is causal. A lower raw-feature cosine loss is still not by itself proof of better
input preservation: historical raw absolute-target runs demonstrated that a shared positional
template can buy a very low reconstruction number.

---

## 1. The problem this design is solving

### 1.1 Current information path

For V-JEPA2-L, the current bottleneck performs:

```text
frozen e                         (B, 1024, 1024)
fixed ZCA, in recent runs        (B, 1024, 1024)
in_proj: 1024 -> 256             (B, 1024,  256)
2 x spatial ConvNeXt + pos       (B, 1024,  256)
to_kv: 256 -> 256                (B, 1024,  256)
3 x read/compete/refine          (B,   32,  256)
final LayerNorm                  (B,   32,  256)
```

There are 1,048,576 detailed scalars and 8,192 final abstract scalars per clip, a 128:1 scalar
ratio. The ratio is not a formal bit-rate theorem, but it correctly identifies a severe continuous
many-to-one map.

### 1.2 The two hard losses of degrees of freedom

The first hard loss is per-token channel projection. A rank-at-most-256 linear map from 1,024 to
256 has a null space of at least 768 dimensions **for every token**. Those components disappear
before the model can use spatial or temporal context to decide whether they matter.

The second hard loss is the 1,024-to-32 attention aggregation. Even with input-dependent attention
weights and three nonlinear reads, 262,144 projected memory scalars become 8,192 slot scalars.
Many different token fields must map to the same slot tensor.

The proposed design cannot eliminate the final 8,192-scalar bottleneck without changing the
project's abstract-state contract. Its purpose is narrower and more principled: let the model see,
mix, and select from richer V-JEPA evidence before it is forced to choose the 8,192 values that
survive.

### 1.3 Why the observed floor points here

The relevant completed evidence is:

- the `lambda_recon=1.0` run improved final present reconstruction by only about `0.0073` versus
  the historical 0.05-weight run, while most of the learned improvement still survived a rolled
  within-source code;
- increasing `N_c` from 32 to 128 improved late training reconstruction by only `0.01405`
  (`2.08%`) and failed the registered representation-health guard;
- the exact EGO4D whitening artifact changes raw pooled channel effective rank from about
  `230/1024` to approximately `1023/1024` before the rank-256 projection;
- optimization was stable: the completed runs do not support NaNs, skipped updates, decoder AGC,
  or a hard-coded loss clamp as the leading explanation.

This leaves whitening, channel width/projection timing, the final `32 x 256` rate, decoder form,
and objective honesty as the live mechanisms. The proposed changes address the first two without
changing the final abstract interface.

### 1.4 The `0.607` warning

The desired “below 0.607” bar must be interpreted within one target geometry. Whitened cosine,
raw cosine, residual-target cosine, and historical relative MSE are not the same measurement. In
particular:

```text
L_recon = 1 - cosine
```

is an angular distortion, not a percentage of information retained. Removing whitening may move
the number below 0.607 simply because dominant/shared raw V-JEPA directions become easy again.
That is useful evidence about rate-distortion, but the preservation claim additionally requires
the correct code to beat a source-diverse shuffled code.

---

## 2. Constraints: what must remain unchanged

The design should preserve the following external facts unless a later experiment explicitly
opens another axis:

| Contract | Preserve it because |
|---|---|
| Frozen selected encoder | The target representation must not move to make reconstruction easy. |
| `Bottleneck.forward(e) -> c` | This is the clean module seam used by training, EMA, drift probes, and tests. |
| `c.shape == (B, N_c=32, D_c=256)` | `F_c`, `Decoder`, losses, diagnostics, and future phases all consume this interface. |
| Three latent read/compete/refine blocks | Depth is not the axis under study. |
| Eight sharpened cosine-attention heads | Attention mechanism is not the axis under study; 512 and 1,024 remain divisible by eight. |
| Orthogonal learned slot identities | Query initialization is not being re-litigated in this experiment. |
| Zero-init residual output projections | Keep the established stable identity-at-init policy; width/projection timing is the intended delta. |
| Decoder size and structure | A larger decoder cannot restore absent code information; hold the recent 512-by-4 recipe fixed for attribution. |
| `N_c=32`, external `D_c=256` | The completed slot sweep rejected more learned query slots as the leading lever. |
| Reconstruction/geometry weights and schedule | Hold them fixed within each comparison so width is not confounded with pressure. |
| Gradient routing and detach rules | Present reconstruction must continue to train only online B and D, never E, `B_EMA`, or `F_c`. |

The key module-design principle is that complexity stays inside `Bottleneck`. Callers continue to
know only the resolved detailed shape and external abstract shape. Internal width is a performance
and capacity characteristic of that module, not a new tensor contract propagated through the
repository.

---

## 3. Change 1 — remove whitening

### 3.1 What changes

The current recent recipe calls `FeatureWhitener.whiten(e)` immediately after the frozen encoder,
before either the online or EMA bottleneck. Removing whitening means both the bottleneck input and
the reconstruction target remain in the selected encoder's raw feature space:

```text
current recent run
    raw e -> fixed ZCA -> B -> c -> D -> reconstructed whitened e

proposed no-whitening arm
    raw e             -> B -> c -> D -> reconstructed raw e
```

This is a **run-configuration change, not an architecture-code change**. Do not delete
`FeatureWhitener`; old checkpoints and intentional whitened experiments still need it. The new run
simply omits `--whiten-features`, `--whiten-stats-path`, and whitening-artifact expectations.

### 3.2 Exact code seam

No model edit is required. `train._present_forward` and `train._coarse_forward` already accept an
optional whitener and apply it only when `cfg.train.whiten_features` is true. The affected runtime
seam is:

```text
train.py
    cfg.train.whiten_features = False
    step_whitener = None
    detailed = encoder(context_clip)       # remains raw
    abstract = bottleneck(detailed)
    reconstruction target = detailed       # remains raw
```

Everything downstream receives the same shapes and dtypes. `FeatureWhitener`, `whiten_stats.py`,
strict artifact validation, and checkpoint support remain untouched for other runs.

### 3.3 Logical effect

Whitening is affine and full-rank here, so it does not itself delete dimensions. It changes what
the finite bottleneck is asked to care about. The raw EGO4D V-JEPA channel cloud is strongly
anisotropic: a few directions carry much more energy than a long tail. Full ZCA raises almost all
1,024 directions to roughly unit variance, so a 256-D projection can no longer prioritize the
directions that V-JEPA naturally made dominant.

Removing ZCA restores the raw rate-distortion problem. A rank-256 observation is then allowed to
spend capacity mainly on dominant, correlated V-JEPA structure rather than treating the long tail
as equally important.

### 3.4 Intuitive effect

Whitening currently says, “every faint detail and every loud detail deserves approximately the
same volume.” The bottleneck then receives only 256 channels and has to cover all of them. Removing
whitening lets it hear V-JEPA's original volume hierarchy and spend its limited capacity on the
signals V-JEPA itself emphasized.

### 3.5 Architectural effect

There is no shape or parameter change. The following do **not** change:

- `in_proj`, ConvNeXt, attention, queries, latent blocks, final `c`, and decoder;
- optimizer parameter groups, AGC, gradient clipping, or EMA updates;
- external `D_c=256`, `F_c`, or checkpoint module keys.

What does change is the coordinate system and conditioning seen by every trainable latent module.
A whitened checkpoint must therefore not be resumed into an unwhitened run. The provenance system
should record `whiten_features=false`, and a fresh initialization is required for a causal arm.

### 3.6 Expected benefit

The most likely immediate observation is a substantially lower raw cosine reconstruction loss,
potentially below 0.607, because the target becomes more compressible and common high-energy
directions are rewarded again. The early 1,024-to-256 projection may become much less damaging in
variance terms because raw channel entropy rank is near 230 rather than 1,023.

It may also improve optimization speed by removing the dense 1,024-by-1,024 whitening matmul and
the fp32-to-bf16 round trip at the encoder seam.

### 3.7 Main risk: a lower number without more input information

Raw absolute V-JEPA features contain strong shared and position-specific structure. Historical raw
absolute-target runs achieved much lower reconstruction loss while relying heavily on a common
template. Therefore:

```text
lower raw L_recon alone != better input-dependent code
```

The correct interpretation requires:

- source-diverse validation examples, not adjacent chunks from one EGO4D source;
- `L_recon_shuffled_c - L_recon_present` increasing in absolute and conditioned-share terms;
- healthy cross-example std/cosine alongside reconstruction;
- no claim that a raw 0.55 is numerically superior to a whitened 0.67 without stating the changed
  target geometry.

Do not add `--recon-residual-target` in the first no-whitening arm. Residual targets may ultimately
be the honest recipe, but adding them simultaneously would prevent attribution to whitening.

---

## 4. Change 2 — expand the complete internal memory width

### 4.1 What “internal memory width” must mean

For this proposal, internal width `M` means the channel width used by **all** of the following:

1. projected detailed tokens;
2. both ConvNeXt spatial mixers;
3. learned memory position embeddings;
4. `to_kv` memory tokens;
5. learned slot queries;
6. cross-attention q/k/v/output projections;
7. slot self-attention;
8. latent-block MLPs;
9. the latent stream immediately before the final abstract projection.

If only items 1–3 are widened while item 4 immediately maps to 256, the experiment has not
expanded attention memory. That narrower option is deliberately rejected as the primary design.

### 4.2 Why the existing config field is the right interface

Before this implementation, `ModelConfig.bottleneck_mixer_dim` controlled `in_proj`, the ConvNeXt
blocks, and `pos_emb`.
No historical paid run could vary it from the CLI, and repository search finds no recorded
non-default experiment. The smallest interface change is to deepen the meaning of this existing
field: it becomes the bottleneck's complete internal width while `d_c` continues to mean only the
external abstract width.

This avoids introducing overlapping `mixer_dim`, `memory_dim`, and `latent_dim` knobs that must be
kept mutually compatible by every caller. One width knob buys more leverage and keeps the internal
shape relationship local to `Bottleneck`.

### 4.3 Current versus proposed construction

Current construction:

```python
mix = cfg.bottleneck_mixer_dim              # 256
in_proj = Linear(D_e, mix)                  # 1024 -> 256
mixers = ConvNeXt(mix)
pos_emb = (1, N_e, mix)
to_kv = Linear(mix, D_c)                    # -> 256
queries = (N_c, D_c)                        # 32 x 256
latent_blocks = BottleneckLatentBlock(D_c)  # 256-wide
norm = LayerNorm(D_c)
```

Proposed construction:

```python
internal = cfg.bottleneck_mixer_dim         # 256, 512, or 1024
in_proj = Linear(D_e, internal)
mixers = ConvNeXt(internal)
pos_emb = (1, N_e, internal)
to_kv = Linear(internal, internal)          # square memory transform
queries = (N_c, internal)
latent_blocks = BottleneckLatentBlock(internal)
abstract_proj = Identity() if internal == D_c else Linear(internal, D_c)
norm = LayerNorm(D_c)                       # remains the final external normalization
```

At the default `internal=D_c=256`, `to_kv`, query shapes, latent-block shapes, `norm`, and the
forward computation remain exactly the current implementation. The new `abstract_proj` is a
parameterless identity, so old default-width state dictionaries do not acquire missing parameter
keys.

### 4.4 The 512-D option

The 512-D path becomes:

```text
e                     (B, 1024, 1024)
in_proj                (B, 1024,  512)
wide ConvNeXt/memory   (B, 1024,  512)
three wide slot reads  (B,   32,  512)
late abstract project  (B,   32,  256)
```

Properties:

- the first linear null space falls from at least 768 to at least 512 dimensions per token;
- attention keys and values carry twice the current channel width;
- the slot processor has 16,384 internal scalars before selecting the final 8,192;
- with eight heads, head dimension rises from 32 to 64;
- cosine attention still normalizes q/k per head, so no new `1/sqrt(d)` calibration is required;
- the final external state and every downstream consumer remain 256-D.

The proposed bottleneck has approximately **18,324,739 parameters**, versus the current
4,837,123, or about **3.79x**. One `(B=64, N_e=1024, M)` bf16 memory tensor rises from about
32 MiB to 64 MiB, while the linear/MLP/pointwise-convolution core has approximately four times the
quadratic work of the 256-D version.

This is the best first engineering arm. It meaningfully delays and softens channel destruction
without paying the full 1,024-D cost. After removing whitening, 512 may already be comfortably
above the raw channel entropy scale, although entropy rank does not prove that all predictive or
token-specific information lives in the dominant subspace.

### 4.5 The 1,024-D option

The 1,024-D path becomes:

```text
e                     (B, 1024, 1024)
square in_proj         (B, 1024, 1024)
wide ConvNeXt/memory   (B, 1024, 1024)
three wide slot reads  (B,   32, 1024)
late abstract project  (B,   32,  256)
```

Properties:

- there is no structurally forced channel-rank reduction before global attention, provided the
  learned square transforms remain full-rank;
- each attention head carries 128 channels;
- the latent processor can use the complete V-JEPA channel width while deciding what to pack into
  the final state;
- this is the literal realization of “only down-project channels at the end.”

The proposed bottleneck has approximately **70,727,427 parameters**, about **14.62x** the current
bottleneck. One bf16 detailed-memory tensor at batch 64 is about 128 MiB, and the quadratic
linear/MLP core is roughly sixteen times the 256-D cost. The frozen encoder may still dominate
some end-to-end costs, but this is no longer a small trainable bottleneck and throughput must be
treated as a first-class performance characteristic.

This is the clean scientific upper bound. If 1,024 materially outperforms 512 with better
correct-code dependence, then the partial 1,024-to-512 projection was still binding. If 1,024 is
flat versus 512, carrying the remaining channels is not worth the cost under this objective.

### 4.6 Initialization policy

Do not change `in_proj` initialization, query initialization, attention temperature, or zero-init
residual gates in the first width experiment. A square 1,024-by-1,024 `in_proj` is almost surely
full-rank at ordinary initialization, so it has no forced algebraic null space even though it may
be imperfectly conditioned. Changing it to identity/PCA/orthogonal initialization at the same time
would add a fourth causal axis.

The new final `abstract_proj` should be nonzero and full-row-rank at initialization. Orthogonal
row initialization with zero bias is a sensible implementation choice because it avoids creating
an accidentally ill-conditioned final map, but it must be documented as part of the new module.
It must **not** be zero-initialized: doing so would erase the orthogonal slot identities and make
the initial external `c` completely identical across slots.

### 4.7 What wider internals do not solve

Wider internals do not change the final 8,192-scalar rate. They therefore cannot guarantee
lossless reconstruction of arbitrary V-JEPA tensors. They also do not directly fix:

- all videos sharing fixed learned query identities at initialization;
- attention slots redundantly reading the same token regions;
- the decoder's ability to exploit constant slot identity plus fixed positions;
- cosine loss ignoring feature magnitude;
- geometry objectives potentially competing with the easiest reconstruction code;
- a final `32 x 256` state that may simply be too small for exact detail.

The hypothesis is not “more hidden parameters magically store more information.” It is “a richer,
input-conditioned computation can make a better choice about which information reaches the same
small external code.”

---

## 5. Change 3 — down-project only after latent reading and refinement

### 5.1 What moves

Before the 2026-07-18 change, the dimensional projection to `D_c=256` happened here:

```text
wide/local mixed tokens -> to_kv(..., D_c=256) -> cross-attention -> slots
```

The proposed projection happens here:

```text
wide tokens -> wide memory -> wide cross-attention/slot refinement
            -> abstract_proj(internal, D_c=256) -> final external c
```

This changes the timing of the irreversible channel choice without moving the external
`Bottleneck` seam.

### 5.2 Logical effect

An early fixed projection must choose the same retained 256-dimensional subspace for every token
before it knows what other tokens contain. Any dropped component can survive only if it is
predictable from the fixed retained subspace.

A late projection follows three rounds of input-dependent cross-attention, slot interaction, and
nonlinear MLP refinement. The wide processor can examine the whole clip and learn to pack the
features relevant for this specific input into the 256 output coordinates. The final projection
is still lossy, but the representation reaching it has been reorganized explicitly for that loss.

### 5.3 Intuitive effect

The current system asks every page of a 1,024-page book to discard most of its vocabulary before
the 32 summarizers read it. The proposed system lets the summarizers read a much richer version of
the book, discuss it, and only then write 32 short final reports. The reports remain short, but
their limited words can be allocated using knowledge of the complete book.

### 5.4 Exact forward path

The proposed `Bottleneck.forward` remains one call and one returned tensor:

```python
tokens = self.in_proj(detailed)  # D_e -> internal
grid = reshape_to_time_major_spatial_grid(tokens)
mixed = self.mixers(grid)
mixed = flatten_grid(mixed) + self.pos_emb
memory = self.to_kv(mixed)       # internal -> internal

slots = self.queries.expand(batch, -1, -1)  # N_c x internal
for block in self.latent_blocks:
    slots = block(slots, memory)             # stays internal-wide

abstract = self.norm(self.abstract_proj(slots))
return abstract                                      # N_c x D_c=256
```

For `internal=256`, `abstract_proj` is an identity and this is exactly the current
`LayerNorm(slots)` path. For 512/1,024, the existing final normalization remains on the external
256-D tensor, so `L_var`, covariance, flow noise scale, and decoder inputs do not inherit
width-dependent output magnitudes.

### 5.5 Gradient routing

Present reconstruction gradients become:

```text
L_recon
  -> Decoder
  -> external c (32 x 256)
  -> final LayerNorm(D_c)
  -> abstract_proj (M -> 256)
  -> wide latent blocks / queries
  -> wide memory / ConvNeXt / in_proj
```

Nothing reaches the frozen encoder because its output remains a no-grad target/input. Nothing
reaches `F_c` in present-only mode. `B_EMA` remains a frozen deepcopy updated only by EMA. In full
prediction mode, both online and EMA bottlenecks still emit `D_c=256`, so `F_c` requires no edit.

`L_var`, `L_cov`, SIGReg, slot metrics, drift probes, and reconstruction diagnostics continue to
observe only the external `c`. That is desirable: their interpretation and dimension remain
stable while the internal compressor changes.

### 5.6 Why a separate low-dimensional key / high-dimensional value mechanism is deferred

Another plausible design would keep narrow attention keys but wider values. It could reduce
compute, but it requires changing `SharpCrossAttention` to accept different query, key, value, and
output dimensions, deciding where each head is projected, and testing a new attention interface.
That is a larger mechanism change than the requested width/timing experiment.

Using one internal width for memory and slots lets the existing sharpened cosine attention remain
unchanged. Separate key/value widths should be considered only if 1,024-wide memory is informative
but too slow—not introduced preemptively.

---

## 6. Exact minimal code surface

### 6.1 `config.py`

No new dataclass field is required.

- Update the `bottleneck_mixer_dim` comment to define it as the complete internal memory/slot width.
- Keep its shipped default at 256.
- Keep `d_c=256` as the external abstract width.
- Validate positive width and divisibility by `bottleneck_cross_attn_heads` before model creation.

Using one width field prevents invalid states such as a 512-D mixer, 768-D memory, and 256-D slot
processor that every caller must understand.

### 6.2 `models.py`

All architectural changes remain inside `Bottleneck.__init__` and `Bottleneck.forward`:

1. change `to_kv` from `Linear(mix, d_c)` to `Linear(mix, mix)`;
2. change learned queries from `(n_c, d_c)` to `(n_c, mix)`;
3. construct every `BottleneckLatentBlock` at `mix`;
4. add conditional `abstract_proj` after latent processing and immediately before the existing
   `LayerNorm(d_c)`;
5. preserve the returned `(B, n_c, d_c)` tensor and current attention-map shape.

`SharpCrossAttention`, `BottleneckLatentBlock`, `ConvNeXtBlock`, `TargetBottleneck`, `Decoder`, and
`CoarseFlow` do not require implementation changes.

### 6.3 `train.py`

- Add one architecture CLI flag, `--bottleneck-mixer-dim`, mapped to the existing config field
  before module construction.
- Describe it as an internal-memory/slot width and state that checkpoints are incompatible across
  non-default values.
- Add early validation for width/head compatibility.
- Do not add a separate “late projection” flag; late projection is the internally coherent meaning
  of `mix > d_c`, and the `mix=d_c` default remains the exact legacy path.
- Removing whitening remains command configuration only; no train-step branch changes.

The existing resource/no-step provenance machinery automatically serializes the dataclass field.

### 6.4 Optimizer and AGC

No grouping code needs a special case.

- `abstract_proj.weight` is an ordinary content-bearing linear matrix, so it should receive normal
  weight decay and bottleneck AGC.
- its bias automatically enters the no-decay/no-AGC group through the existing shared predicate;
- queries, position embeddings, and zero-init latent outputs retain their existing exclusions;
- the optimizer already iterates every trainable bottleneck parameter.

### 6.5 Checkpoint and probe compatibility

- Existing 256-wide checkpoints rebuild `mix=256`; the conditional projection is an identity with
  no state and the existing final norm is unchanged, so their B state keys can remain compatible.
- 512/1,024 checkpoints are shape-incompatible with 256 and must start fresh; normal resume is valid
  only with the same saved config.
- `TargetBottleneck` is a deepcopy, so the new parameters automatically participate in EMA without
  new routing logic.
- checkpoint config serialization already includes `bottleneck_mixer_dim`.
- `drift_probe.model_config_from_checkpoint_dict` filters against `ModelConfig`, so it already
  restores the field and will build the correct B when code understands the new semantics.
- dataset, encoder, and whitening provenance remain independent identities.

### 6.6 Tests required before implementation is considered complete

Tests should cross the existing `Bottleneck.forward` interface rather than exposing a second public
interface for internal tensors.

1. **Default parity test:** with `mix=d_c`, verify the current output formula, state keys, attention
   shape, and identity-at-init behavior remain unchanged.
2. **Wide shape test:** use a small fixture where `D_e > mix > d_c`; verify memory-compatible
   construction and returned `(B,n_c,d_c)`.
3. **Wide identity-at-init test:** two different inputs must still produce the same initial output,
   while different slot identities survive the final nonzero projection/normalization.
4. **Bridge-opening test:** after one optimizer step, gradients must reach the new final projection;
   after the zero bridge opens, gradients must reach cross-attention q/k/v and `in_proj`.
5. **Reconstruction gradient contract:** `D(B(e))` trains D and every wide-B segment, not `F_c`,
   E, or `B_EMA`.
6. **EMA equality/update test:** fresh B and `B_EMA` match at the external 256-D output, and EMA
   updates the new final projection.
7. **Optimizer grouping test:** final projection weight decays/receives AGC; bias/norm do not.
8. **Checkpoint test:** old 256-wide B state loads; wide checkpoint round-trips; width mismatch is
   rejected before model mutation.
9. **CLI/provenance test:** the requested internal width appears in resolved config and provenance.
10. **Resource preflight:** measure 512 and 1,024 at the paid physical batch before a full launch;
    throughput, not only OOM, is an acceptance characteristic.

After implementation, the living-reference sections in `AGENT_FILES/AGENTS.md`, the architecture
flag table, and relevant code/metric guides must be updated in the same session.

### 6.7 Files deliberately untouched

The following should receive no functional edit for this mechanism:

- `encoders.py` and encoder adapters;
- `data.py` and EGO4D sampling;
- `losses.py` reconstruction/geometry formulas;
- `diagnostics.py` representation and shuffled-code formulas;
- `Decoder` and `CoarseFlow` architecture;
- whitening implementation and offline-stat tooling;
- EMA/detach rules in the train step.

If implementation pressure spreads into those files, the change is probably exceeding the
intended seam.

---

## 7. How the three changes transform the model

### 7.1 Logical transformation

The current model makes a global, fixed channel decision first and a video-dependent token decision
second. The proposed model restores the natural feature geometry, allows a larger video-dependent
read/refinement computation, and makes the narrow external channel decision last.

```text
current logic
    standardize every channel equally
    -> discard channels independently per token
    -> reason over the survivors

proposed logic
    preserve V-JEPA's natural importance hierarchy
    -> reason over a richer token field
    -> choose the final abstract coordinates after global evidence is available
```

### 7.2 Intuitive transformation

The change does not make the final notebook larger; it gives the note-taker more complete source
material and more scratch space before writing the same 32 short notes. That is why it may improve
preservation without weakening the final abstract-state contract.

### 7.3 Architectural transformation

The bottleneck becomes a deeper module: its external interface stays simple and stable, while
memory width, latent width, late compression, and output normalization are implemented internally.
No downstream module learns an internal-width concept.

### 7.4 Information-theoretic restraint

Moving the projection cannot make `32 x 256` lossless. It changes the **allocation** of the fixed
rate, not the rate itself. If the final code is below the intrinsic dimension of the clip-specific
V-JEPA target, the loss will still plateau. A flat 1,024-wide result would be evidence that the
final code, decoder/objective, or target unpredictability—not the timing of channel projection—is
binding.

---

## 8. Original attribution-first experimental order (superseded for this launch)

This was the original recommendation before the 2026-07-18 execution decision. It is preserved to
make the attribution tradeoff auditable, but it is not the launch order in the active guide.

| Arm | Whitening | Internal width | Projection timing | What it answers |
|---|---|---:|---|---|
| Existing control | full ZCA | 256 | effectively already 256 | Current clean-commit floor (`N_c=32` sweep control). |
| A | **none** | 256 | effectively already 256 | Does full ZCA create the difficult/common numerical regime? |
| B | none | **512** | `512 -> 256` after latent blocks | Does a practical wide-read/late-write path improve correct-code reconstruction? |
| C | none | **1,024** | `1024 -> 256` after latent blocks | Does eliminating all forced early channel-rank reduction beat 512 enough to justify cost? |

Hold the following fixed across A/B/C:

- EGO4D data identity, encoder revision, seed, batch, steps, and transform recipe;
- `N_c=32`, external `D_c=256`, latent depth 3, and eight heads;
- decoder width/depth from the current clean reconstruction recipe;
- present-only absolute cosine target, `lambda_recon=1`, variance/covariance weights, warmup, LR,
  clipping, and schedule;
- source-diverse diagnostic batch construction once that measurement contract is repaired.

This sequence costs two new architecture arms after the config-only no-whitening arm and gives a
clean chain of comparisons:

```text
control -> A isolates whitening
A -> B isolates wide internal processing plus late projection
B -> C isolates the remaining 512 -> 1024 internal-width difference
```

Changes 2 and 3 can be separated only by adding an artificial “wide ConvNeXt but early
`to_kv(...,256)`” arm. That arm answers a narrower implementation question and spends a paid run
on a design we do not want. Given the stated bandwidth constraint, treat wide memory and late
projection as one coherent mechanism.

If only two new paid runs are possible, run A and B. If only one new paid run is possible, run A:
it is config-only, directly tests the strongest current suspect, and determines whether widening
is still warranted in raw feature space.

---

## 9. What to expect and how to interpret it

### 9.1 Expected trajectories

**No whitening, 256 internal:** likely lower raw reconstruction loss and faster early progress. The
main uncertainty is whether the correct-code gap improves or the decoder merely recovers an easier
shared raw template.

**No whitening, 512 internal/late projection:** likely a further but smaller reconstruction gain if
the current per-token channel squeeze is binding. A scientifically useful win must accompany
greater correct-code dependence and healthy external-code geometry.

**No whitening, 1,024 internal/late projection:** should be the best of the three if early channel
loss is genuinely causal. It may converge more slowly in wall-clock time because its trainable
quadratic core is much larger. A tiny gain over 512 argues strongly for keeping 512.

### 9.2 Outcome matrix

| Observation | Interpretation |
|---|---|
| Raw loss drops below 0.607 and shuffled-code gap grows materially | Removing whitening/widening improved recoverable input-specific content. |
| Raw loss drops but shuffled-code gap stays near zero | The target became easier through shared/template structure; preservation did not improve proportionally. |
| Loss changes little but shuffled-code gap grows | The code is more input-dependent even though the fixed final rate still controls angular distortion. This is a preservation win, not a floor win. |
| 512 beats 256; 1,024 beats 512 | Early channel width is genuinely binding; choose cost/quality point from the curve. |
| 512 beats 256; 1,024 is flat | 512 is sufficient internal width; do not pay for 1,024. |
| Neither width beats unwhitened 256 | Projection timing is not the leading limit in raw space; inspect final rate, decoder form, or objective honesty. |
| Wider arms lower loss but harm std/cosine or correct-code gap | Extra internal capacity is being spent on fixed slot/template geometry rather than video information. |
| 1,024 destabilizes while 512 is healthy | The mechanism may work but the 1,024 optimization/cost point is impractical under the unchanged schedule. |

### 9.3 Success is joint, not a single loss threshold

The primary scientific question is whether more of the reconstruction improvement depends on the
correct input code. Therefore success requires all of:

1. stable training with no skipped/nonfinite updates;
2. a lower late-window reconstruction value **within the same target geometry**;
3. a larger source-diverse shuffled-code gap/conditioned share;
4. healthy cross-example spread and cosine on a source-diverse batch;
5. no evidence that rank is supplied mostly by fixed slot identity;
6. a cost/throughput point compatible with later full-prediction training.

Raw loss is necessary for the floor question but insufficient for the preservation question.

### 9.4 Expected impact on the other known Phase-1 problems

| Existing problem | Expected effect of these changes |
|---|---|
| Weak exact-input dependence / template decoding | No whitening can make this worse by restoring shared raw directions; wide/late processing can make it better only if the new capacity is used for clip content. The shuffled-code gap remains decisive. |
| Fixed, input-independent slots at initialization | Unchanged. Orthogonal queries and zero residual outputs remain, so every clip still receives the same initial scaffold. This is intentionally not bundled into the width experiment. |
| Slot redundancy / repeated attention coverage | Not directly solved. Wider values carry more content, but all slots can still attend to the same tokens because attention is normalized over memory independently per slot. |
| Low or misleading effective rank | External `c` remains 256-D and geometry losses are unchanged. Wider internal rank does not guarantee higher content rank; fixed slot identity can still inflate pooled rank. |
| Weak future prediction / copy-baseline failure | Not measured by present-only reconstruction. A richer `c` may later help because it contains more state, or hurt because it becomes more temporally variable and harder for `F_c` to predict. Transfer to full prediction needs a separate run. |
| Reconstruction blindness to feature norm | Unchanged. Cosine loss still ignores target/prediction magnitude, and final LayerNorm still removes slot-level scale information. |
| Early zero-gate/schedule bias | Unchanged. Wider q/k/v paths still open through zero-init output projections while reconstruction ramps and geometry losses are active. |
| Optimization instability | Removing whitening removes a large fp32 transform, but 512/1,024 add substantially larger trainable matrices and gradients. Stability cannot be inferred from parameter count; monitor the existing skip/AGC/norm signals. |

The present design is therefore a targeted rate-allocation change, not a universal collapse or
prediction fix. Its success would justify carrying the improved compressor into the next
full-prediction experiment; it would not itself license a forecasting claim.

---

## 10. Risks and mitigations

### Risk 1 — no whitening rewards the old template shortcut

**Mechanism:** dominant raw and per-position feature directions can be reconstructed from fixed slot
identity and decoder position information.

**Mitigation:** keep the first arm absolute for attribution, but require source-diverse shuffled
evaluation. If loss falls without code dependence, the next objective arm should use the existing
per-position residual target rather than pretending the raw number is success.

### Risk 2 — wider learned queries increase fixed scaffold capacity

**Mechanism:** 32 orthogonal queries at width 512/1,024 provide a larger input-independent internal
scaffold before the final projection.

**Mitigation:** external `c` remains 256-D and the final projection is shared, but diagnostics must
compare real-input `c` with zero-input/mean/shuffled controls. Do not infer content from pooled rank
alone.

### Risk 3 — zero-init content path opens slowly relative to geometry losses

**Mechanism:** the widened q/k/v and memory weights receive useful content gradients only after the
zero output bridges move; reconstruction still ramps while geometry is active.

**Mitigation:** retain the schedule for the first causal width comparison. If both wide arms stay
input-independent despite healthy optimization, instrument `||B(e)-B(0)||/||B(e)||` before changing
initialization or schedules.

### Risk 4 — 1,024 width is computationally disproportionate

**Mechanism:** ConvNeXt pointwise layers, attention projections, slot self-attention, and latent
MLPs all contain width-squared matrices.

**Mitigation:** perform a real resource/throughput preflight; use 512 as the default practical arm;
consider asymmetric keys/values only after evidence that 1,024 content width helps.

### Risk 5 — the final projection simply becomes the new choke point

**Mechanism:** `abstract_proj` still maps each 512/1,024-D slot to 256 dimensions.

**Mitigation:** this is intentional—the project requires a narrow external state. If wide internal
processing helps but plateaus, the remaining question is external `D_c`, not another hidden-width
increase. That would require explicit approval because it changes downstream architecture.

### Risk 6 — raw and whitened loss values are compared as if identical

**Mechanism:** changing target covariance changes the natural cosine rate-distortion scale.

**Mitigation:** compare learning improvement, correct-vs-shuffled separation, and geometry within
each coordinate regime; label cross-regime raw numbers observationally, not as equal-unit scores.

---

## 11. Alternatives deliberately not included

The following ideas may be valuable but would make this experiment uninterpretable if bundled now:

- PCA/orthogonal/identity initialization of `in_proj`;
- input-derived slot seeding or changing zero-init residual gates;
- balanced Slot Attention or structured local slot assignments;
- separate low-dimensional keys and wide values;
- changing `N_c`, external `D_c`, decoder capacity, or decoder topology;
- changing cosine loss to a cosine-plus-MSE objective;
- enabling residual reconstruction targets in the first no-whitening arm;
- changing geometry weights, warmup order, LR, or training duration.

These are follow-up mechanisms, not prerequisites. The current proposal has a clean question:

> Does restoring raw V-JEPA geometry and delaying channel compression until after a 512/1,024-D
> input-dependent latent processor improve the amount of clip-specific detail recoverable from the
> same `32 x 256` abstract state?

---

## 12. Final recommendation

Implement the width/projection change entirely inside `Bottleneck`, keep the default 256-D path
state-compatible and computationally identical, and expose the existing mixer field through one
CLI flag. Do not touch the decoder, flow, losses, detach boundaries, or whitening implementation.

The active launch now runs unwhitened 512 and 1,024 sequentially, both with late projection and all
auxiliary regularizer weights at zero. The earlier unwhitened-256-first recommendation is retained
in Section 8 only as the cleaner attribution design that the human chose not to fund in this bundle.

If all three arms are executed, interpret them as:

```text
no whitening      -> restores natural target compressibility
wide internals    -> gives the compressor richer evidence and scratch space
late projection   -> lets global, input-dependent reasoning choose what the final 256 channels keep
```

This is the most surgical form of the proposal: the abstract state remains genuinely small, while
the architecture stops destroying three quarters of every token's channels before it has had a
chance to understand the clip.
