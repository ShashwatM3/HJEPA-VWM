# 00 — Truth map and notation

This chapter defines the language used by the rest of the corpus. Learn it first: most apparent
contradictions in this project come from mixing a tensor with its model, a shipped default with an
experiment override, or the current Phase 1 implementation with the future hierarchy.

## Four layers of truth

| Label | Exact meaning | Example |
|---|---|---|
| **IMPLEMENTED** | Executable in the audited working tree | `CoarseFlow`, feature reconstruction, strict resume |
| **SHIPPED DEFAULT** | Value returned by the config dataclasses before CLI overrides | `N_c=32`, `D_c=256`, `M=256`, `k=4` |
| **EXPERIMENT RECIPE** | An intentional override in a specific run or investigation | Investigation 017 uses `M=512` and present-only training |
| **PLANNED** | A target architecture that is not executable here | `F_e`, a VAE-latent frame generator, horizon conditioning |

“The architecture uses `M=512`” is therefore incomplete. The correct statement is: “the shipped
default bottleneck width is 256; Investigation 017 overrides it to 512.” The same discipline applies
to every number in this course.

## The central objects

| Symbol | Name | Default shape | Owner |
|---|---|---:|---|
| `x` | raw RGB clip | `(B, T=8, 3, 256, 256)` | data pipeline |
| `e` | detailed frozen-encoder tokens | `(B, N_e, D_e)` | `FrozenVideoEncoder` |
| `c` | abstract latent slots | `(B, N_c=32, D_c=256)` | online `Bottleneck` |
| `c⁺` | future abstract target | `(B, 32, 256)` | EMA target bottleneck |
| `ε` | Gaussian flow source | `(B, 32, 256)` | training step |
| `τ` | flow time | `(B, 1, 1)` | training step |
| `z_τ` | interpolated flow state | `(B, 32, 256)` | training step |
| `u` | target velocity | `(B, 32, 256)` | flow-matching target |
| `û_θ` | predicted velocity | `(B, 32, 256)` | `CoarseFlow` |
| `ê` | reconstructed detailed features | `(B, N_e, D_e)` | feature `Decoder` |

The plus sign in `c⁺` means “future target,” not addition. A hat means prediction. The superscript
is semantic; it does not introduce a new tensor axis.

## Model names versus tensor names

The project deliberately uses short mathematical names:

- `E`: one frozen pretrained visual encoder. It produces detailed features `e`.
- `B`: the trainable online bottleneck. It produces abstract slots `c`.
- `B_EMA`: a frozen exponential-moving-average copy of `B`. It produces target slots `c⁺`.
- `F_c`: the coarse conditional flow model. It predicts a velocity in abstract-latent space.
- `D`: the current feature reconstruction decoder. It reconstructs `e`, not pixels.
- `F_e`: the planned fine flow model. It does not exist in the current code.
- `G`: a planned frame/VAE-latent generator. It does not exist in the current code.

The executable model tuple is exactly:

```text
(encoder, bottleneck, target_bottleneck, coarse_flow, decoder)
```

`FeatureMeanTracker` and `FeatureWhitener` are stateful buffers outside that tuple. They have no
trainable parameters.

## Axis notation

| Axis | Meaning | Default |
|---|---|---:|
| `B` | batch size | 64 |
| `T` | clip frames | 8 |
| `C` | RGB channels | 3 |
| `H,W` | input height and width | 256, 256 |
| `T_e,H_e,W_e` | encoder token lattice | encoder-dependent |
| `N_e` | number of detailed tokens, `T_e H_e W_e` | 1024 or 2048 |
| `D_e` | detailed token width | 1024 or 768 |
| `N_c` | number of abstract slots | 32 |
| `D_c` | abstract slot width | 256 |
| `M` | bottleneck internal width | 256 |
| `D_d` | decoder internal width | 256 |
| `k` | future offset in decoded-frame indices | 4 |
| `s` | within-clip temporal stride | 2 |

`M` is not the number of slots and is not the abstract output width, even though all three shipped
defaults happen to be 256/32/256. Keeping `M`, `N_c`, and `D_c` separate is essential when reading
the Investigation 017 grid.

## Encoder-dependent lattice

| Encoder | `T_e × H_e × W_e` | `N_e` | `D_e` |
|---|---:|---:|---:|
| V-JEPA2 ViT-L | `4 × 16 × 16` | 1024 | 1024 |
| SigLIP2 ViT-B | `8 × 16 × 16` | 2048 | 768 |
| DINOv3 ViT-B | `8 × 16 × 16` | 2048 | 768 |

V-JEPA2 uses a two-frame tubelet, hence eight input frames become four temporal token planes.
SigLIP2 and DINOv3 are applied per frame, so they retain eight token planes. This is why the same
abstract bottleneck can face either 1024 or 2048 detailed tokens.

## Temporal vocabulary

The clip frame indices are:

```text
context: start + [0, 2, 4, 6, 8, 10, 12, 14]
target:  start + k + [0, 2, 4, 6, 8, 10, 12, 14]
```

These are decoded-video frame indices, not seconds and not encoder-token indices. With default
`k=4`, the two clips share six of their eight exact decoded frames. The target is “future shifted,”
but it is not disjoint.

## Three kinds of state

1. **Parameters** receive gradients or are EMA copies of parameters.
2. **Buffers** are checkpointed state without optimizer gradients, such as feature means,
   whitening matrices, and positional encodings.
3. **Runtime state** includes optimizer moments, data-sampler position, Python/NumPy/Torch RNG
   states, W&B run identity, and provenance envelopes.

An airtight explanation of resume must cover all three. Loading only model weights is not exact
resume.

## Five invariants to memorize

1. The pretrained encoder is frozen, absent from the optimizer, and absent from checkpoints.
2. There is no separate target encoder; context and target both use the same frozen `E`.
3. Only the bottleneck has an EMA target copy.
4. The current decoder reconstructs encoder features, never RGB or VAE latents.
5. The current model has no learned future-horizon input; `k` changes the data pair, not the model
   signature.

## Dimensional sanity checks

- Raw clip: `8 × 3 × 256 × 256 = 1,572,864` scalar values.
- V-JEPA2 detailed representation: `1024 × 1024 = 1,048,576` scalars.
- Frame-encoder detailed representation: `2048 × 768 = 1,572,864` scalars.
- Abstract representation: `32 × 256 = 8,192` scalars.
- V-JEPA2 detailed-to-abstract scalar compression: `1,048,576 / 8,192 = 128:1`.
- SigLIP2/DINOv3 detailed-to-abstract scalar compression: `1,572,864 / 8,192 = 192:1`.

Those ratios compare scalar counts, not information content, entropy, or storage after compression.

## Precision vocabulary

- RGB data becomes floating point in `[0,1]` before encoder normalization.
- Frozen encoder compute is configured as `bf16` by default on CUDA.
- The bottleneck, flow, and decoder normally run under CUDA bfloat16 autocast.
- Several numerically sensitive losses and diagnostics explicitly enter FP32.
- Whitening statistics accumulate/eigendecompose in FP64, then store operational buffers in FP32.
- Checkpoint size is not inferable from parameter count alone because optimizer states, dtypes, and
  serializer details matter.

## A quiz-quality one-sentence answer

> Phase 1 freezes a pinned pretrained visual encoder `E`, compresses its spatiotemporal tokens with
> an online Perceiver-like bottleneck `B`, uses an EMA copy of only that bottleneck to define future
> targets, trains a conditional DiT-like flow `F_c` in abstract-slot space, and optionally teaches a
> cross-attention decoder `D` to reconstruct detailed encoder features—while strict provenance,
> deterministic sampling, diagnostics, and exact-resume state make the experiment operable.
