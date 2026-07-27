# 03 — Frozen encoder seam

## Why the seam exists

All downstream modules accept one contract:

```text
encode(x: B×T×3×H×W) -> e: B×N_e×D_e
```

An `EncoderSpec` records the exact coordinate system behind `e`: repository, immutable revision,
input size, normalization, token lattice, width, precision, attention implementation, frame
microbatch, parameter count, cache location, and implementation details. A SHA-256 fingerprint of
the full specification protects checkpoints, caches, and whitening artifacts from silent encoder
drift.

## Common invariants

Every adapter:

- validates the raw input shape;
- applies encoder-specific normalization internally;
- uses a pinned Hugging Face repository revision;
- freezes every parameter;
- runs in evaluation mode;
- returns only patch/spatiotemporal tokens in `(B,N_e,D_e)`;
- exposes exact lattice metadata;
- supports FP32, or BF16 on CUDA;
- is absent from the optimizer and checkpoint payload.

Calling `.train()` on the outer training system must not turn a frozen encoder into a stochastic
training-mode model.

## Exact encoder ledger

| Alias/family | Pinned model | Revision | Normalization | Lattice | `N_e×D_e` | Parameters |
|---|---|---|---|---:|---:|---:|
| V-JEPA2 | `facebook/vjepa2-vitl-fpc64-256` | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | ImageNet | `4×16×16` | `1024×1024` | 325,971,328 |
| SigLIP2 | `google/siglip2-base-patch16-256` | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` | mean/std 0.5 | `8×16×16` | `2048×768` | 85,843,200 |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` | ImageNet | `8×16×16` | `2048×768` | 85,660,416 |

The code pins Transformers `4.57.6`. The local validation environment had 5.5.3, which correctly
tripped the repository's pin test; do not “fix” the test by weakening it.

## V-JEPA2 adapter

Input layout is converted to the model's video convention. V-JEPA2 uses a tubelet temporal support
and stride of two, so eight RGB frames produce four temporal planes. Each plane is a 16×16 patch
grid:

```text
T_e = 8 / 2 = 4
H_e = W_e = 256 / 16 = 16
N_e = 4 * 16 * 16 = 1024
D_e = 1024
```

The adapter uses the vision-feature path rather than a classification head. Its 325,971,328
parameters make it much larger than the Phase 1 trainable stack, but freezing it eliminates its
parameter gradients and Adam moments.

FP32 detailed tensor size per video:

```text
1024 * 1024 * 4 = 4 MiB
```

BF16 is 2 MiB per video.

## SigLIP2 adapter

SigLIP2 is a frame encoder. The adapter flattens the batch and time axes:

```text
(B,T,3,256,256) -> (B*T,3,256,256)
```

It processes frames in configurable microbatches, shipped as eight frames per microbatch, then
restores the video axis and flattens frame-patch tokens:

```text
(B*T,256,768) -> (B,T,256,768) -> (B,2048,768)
```

The count 85,843,200 is the retained vision patch tower after removing the 7,087,104-parameter
pooling/head portion not used by this adapter.

Detailed tensor per video:

```text
2048 * 768 = 1,572,864 scalars
FP32 = 6 MiB; BF16 = 3 MiB
```

## DINOv3 adapter

DINOv3 is also frame-based. For each 256×256 frame, the pretrained model exposes:

```text
1 CLS + 4 register + 256 patch = 261 tokens
```

The adapter strips the first five non-patch tokens and retains exactly 256 patch tokens. Eight
frames therefore produce `(B,2048,768)`. Forgetting to remove the class/register tokens would break
the documented spatial lattice and positional assumptions in downstream modules.

## Normalization

The data pipeline emits raw `[0,1]` RGB. The adapter then applies its pinned normalization:

- V-JEPA2 and DINOv3: ImageNet channel statistics;
- SigLIP2: channel mean 0.5 and standard deviation 0.5.

Whitening is not normalization of RGB. It is an optional later operation on `e`.

## Precision and attention

Shipped encoder precision is BF16, valid only on CUDA. Selecting FP32 disables the outer encoder
autocast path. The attention implementation is recorded as SDPA. These choices enter the
fingerprint because a supposedly identical model under a different compute implementation can
produce measurably different features.

Frame encoder microbatching controls peak memory without changing the logical output. It is also
fingerprinted: operational choices that affect execution are part of reproducibility.

## Fingerprint purpose

The fingerprint is used to reject:

- resuming a checkpoint with a different encoder family or revision;
- loading a feature cache generated under another normalization or token layout;
- applying whitening statistics fit to another feature coordinate system;
- claiming two runs are comparable when their frozen representation differs.

The Hugging Face cache directory is included. This is stricter than pure mathematical identity but
helps make the resolved execution context explicit.

## Frozen does not mean cheap

No-gradient execution removes backward activations and encoder optimizer state, but the forward pass
still costs compute and memory. Two windows normally mean two encoder paths. Present-only training
is faster precisely because it avoids target-window encoding and coarse-flow work.

Approximate BF16 output alone at batch 64:

| Encoder | One detailed batch | Context + target |
|---|---:|---:|
| V-JEPA2 | 128 MiB | 256 MiB |
| SigLIP2/DINOv3 | 192 MiB | 384 MiB |

Again, these are output tensors only.

## Why there is no encoder EMA

The original design envisioned a trainable online encoder and an EMA target encoder. Current Phase 1
instead freezes a strong pretrained encoder and studies the bottleneck/predictor interface. Because
`E` never changes, a second encoder copy would be identical and wasteful. The moving target arises
at `B`, so only `B` has an EMA teacher.

## Quiz traps

- **Wrong:** “V-JEPA2 emits eight temporal token planes.” **Right:** its two-frame tubelets emit
  four planes from eight input frames.
- **Wrong:** “DINO returns 261 spatial tokens per frame downstream.” **Right:** the adapter removes
  one class and four register tokens, retaining 256 patches.
- **Wrong:** “All encoders use ImageNet normalization.” **Right:** SigLIP2 uses 0.5/0.5.
- **Wrong:** “The encoder is in the checkpoint because it is part of the model.” **Right:** it is
  reconstructed from a pinned specification; weights are not serialized.
- **Wrong:** “Frozen means no state contract.” **Right:** revision, preprocessing, precision, token
  layout, and implementation are fingerprinted.
