# 04 — Bottleneck internals

## Contract

```text
B: e ∈ R^(B×N_e×D_e) → c ∈ R^(B×N_c×D_c)
```

At shipped defaults, `N_c=32`, `D_c=256`, and internal width `M=256`. The bottleneck is
encoder-aware at construction because it needs `T_e,H_e,W_e,D_e`, but it returns a common abstract
shape for every encoder.

## Stage 1: project and restore the lattice

Each detailed token is projected:

```text
h = e W_in + b_in
(B,N_e,D_e) → (B,N_e,M)
```

The sequence is reshaped to the known token lattice and time is folded into the batch:

```text
(B,T_e,H_e,W_e,M) → (B*T_e,M,H_e,W_e)
```

This makes local mixing spatial within each encoder temporal plane. The ConvNeXt blocks do not mix
time directly.

## Stage 2: two shared ConvNeXt blocks

Each of the two blocks contains:

1. 7×7 depthwise convolution, `groups=M`;
2. channels-last LayerNorm;
3. pointwise MLP `M → 4M`;
4. GELU;
5. pointwise projection `4M → M`;
6. residual addition.

The two blocks share across every temporal plane because time has been folded into the batch.
Afterward, the lattice is flattened back to `(B,N_e,M)`.

## Stage 3: positional encoding and key/value projection

A learned positional table has shape:

```text
(1,N_e,M)
```

It is initialized from a truncated normal with standard deviation 0.5. This is encoder-lattice
specific: a V-JEPA bottleneck has 1024 rows, while a frame-encoder bottleneck has 2048.

The position-enriched features pass through a learned `M→M` key/value preparation projection.

## Stage 4: learned abstract queries

The bottleneck owns `N_c` learned query vectors:

```text
queries: (N_c,M)
```

They are initialized as orthogonal unit rows, then broadcast over the batch. At the start of
training these queries, rather than input content, define the abstract slot identities.

## Stage 5: three latent blocks

Each block has three residual sublayers.

### Sharpened cosine cross-attention

Queries are normalized; keys are normalized; their cosine similarities are multiplied by a learned
positive scale:

```text
logits = cosine(q,k) * min(exp(logit_scale), 100)
```

There are eight heads. The log scale begins at `log(1/0.07)`, so the initial temperature is 0.07
and the scale is approximately 14.2857. Values come from the detailed token stream; the attention
result is projected back to `M`.

The cross-attention output projection is zero-initialized.

### Latent self-attention

Layer-normalized slots attend to one another with PyTorch multi-head attention. The output projection
is zero-initialized.

### Latent MLP

LayerNorm, `M→4M`, GELU, and `4M→M`. The final projection is zero-initialized.

All three outputs enter residual additions.

## Why zero initialization matters

Every input-dependent bridge into the residual slot stream starts at zero. Therefore, at exact
initialization:

- the output is independent of video content;
- every video receives the same normalized learned-query template;
- gradients first open the zero output bridges;
- upstream query/key/value and temperature parameters receive meaningful gradients only as those
  bridges open.

This is an intentional stable-residual initialization, not evidence that a run has already
collapsed. Persistent template behavior after training is collapse.

## Stage 6: abstract projection and normalization

After three latent blocks:

- if `M == D_c`, the abstract projection is `Identity`;
- otherwise it is a learned `M→D_c` linear layer;
- final affine LayerNorm operates at width `D_c`.

Output shape is exactly `(B,N_c,D_c)`.

The final cross-attention map returned for diagnostics has shape:

```text
(B, heads=8, N_c, N_e)
```

For batch 64:

| Encoder lattice | Elements | BF16 storage |
|---|---:|---:|
| `N_e=1024` | 16,777,216 | 32 MiB |
| `N_e=2048` | 33,554,432 | 64 MiB |

This is only the final cross-attention probability tensor.

## Exact parameter formula

Let:

- `C=2` ConvNeXt blocks;
- `L=3` latent blocks;
- `P = M*D_c + D_c` if `M≠D_c`, otherwise zero.

Then:

```text
P_B =
  (D_e*M + M)                    input projection
+ C*(8*M^2 + 57*M)              ConvNeXt blocks
+ (M^2 + M)                     token key/value projection
+ N_e*M                         learned detailed position table
+ N_c*M                         learned abstract queries
+ L*(16*M^2 + 19*M + 1)         latent blocks
+ P                              optional abstract projection
+ 2*D_c                          final affine LayerNorm
```

The `+1` per latent block is the scalar cross-attention logit scale.

## Exact counts

### V-JEPA2 lattice (`N_e=1024,D_e=1024`)

| `M` | `N_c` | `D_c` | Bottleneck parameters |
|---:|---:|---:|---:|
| 256 | 32 | 256 | 4,837,123 |
| 512 | 32 | 256 | 18,324,739 |
| 1024 | 32 | 256 | 70,727,427 |

### Frame-encoder lattice (`N_e=2048,D_e=768`)

| `M` | `N_c` | `D_c` | Bottleneck parameters |
|---:|---:|---:|---:|
| 256 | 32 | 256 | 5,033,731 |
| 512 | 32 | 256 | 18,717,955 |
| 1024 | 32 | 256 | 71,513,859 |

The frame bottleneck is larger at the same `M` because the positional table doubles, even though
its input projection is narrower.

## Scaling intuition

The dominant terms grow as `M²`: two ConvNeXt blocks plus three latent blocks. Doubling `M` from 256
to 512 therefore increases V-JEPA bottleneck parameters from 4.84M to 18.32M, roughly 3.79×.
Doubling again approaches another fourfold increase. `N_c` and `N_e` contribute linearly through
query and position tables, so changing slot count is much cheaper than changing width.

## Slot semantics

Slots are ordered learned queries, not a set with guaranteed permutation invariance. The learned
query table and downstream slot positional embeddings assign stable indices. Slot diversity
diagnostics ask whether different indices encode distinct content, but no loss proves human-readable
object slots.

## What locally mixes what?

- ConvNeXt: neighboring spatial detailed tokens within the same encoder time plane.
- Cross-attention: each abstract slot reads all detailed tokens, including all time planes.
- Latent self-attention: abstract slots exchange information.
- Latent MLP: per-slot channel transformation.

This division is a useful whiteboard answer because it explains where space, time, and slot
interaction enter.

## Common mistakes

- `M` is internal width; `D_c` is output width.
- The learned detailed positional table is not the decoder's fixed sinusoidal position table.
- ConvNeXt does not convolve across encoder time.
- Attention probabilities returned for diagnostics come from the final latent block.
- `B_EMA` has exactly the same architecture/count as `B`; it is a deep copy, not a smaller teacher.
- Zero-initialized residual outputs explain content-independent initialization but do not excuse
  persistent collapse.
