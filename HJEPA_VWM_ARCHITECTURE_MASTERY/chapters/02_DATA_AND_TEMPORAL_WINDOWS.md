# 02 — Data and temporal windows

## Dataset output contract

`VideoPairDataset` returns:

```text
context: (T=8, C=3, H=256, W=256), float in [0,1]
target:  (T=8, C=3, H=256, W=256), float in [0,1]
```

After collation, both become `(B,8,3,256,256)`. The transformation contract is versioned as:

```text
raw-rgb-resize-crop-jitter-v2
```

That string is part of the dataset identity. Changing augmentation semantics without changing the
version would weaken provenance.

## Temporal index equation

For start frame `a`, within-clip stride `s=2`, clip length `T=8`, and horizon `k`:

```text
I_context(i) = a + i*s          for i=0..7
I_target(i)  = a + k + i*s      for i=0..7
```

The context span is `a..a+14`; the target span is `a+k..a+k+14`.

### Exact overlap

When `k` is even, exact shared-frame count is:

```text
max(0, T - k/s)
```

| `k` | Context indices | Target indices | Shared exact frames |
|---:|---|---|---:|
| 0 | `0,2,…,14` | `0,2,…,14` | 8 |
| 2 | `0,2,…,14` | `2,4,…,16` | 7 |
| 4 default | `0,2,…,14` | `4,6,…,18` | 6 |
| 8 | `0,2,…,14` | `8,10,…,22` | 4 |
| 12 | `0,2,…,14` | `12,14,…,26` | 2 |
| 14 | `0,2,…,14` | `14,16,…,28` | 1 |
| 16 | `0,2,…,14` | `16,18,…,30` | 0 |

For odd `k`, the two parity grids have no exact shared indices, although their temporal spans may
overlap. The strict span-disjoint condition is `k > 14`; the project's normal even horizons make
`k=16` the first disjoint choice.

This distinction matters: the default prediction task has strong raw-frame overlap. It can make a
copy-like abstract predictor look better than a truly extrapolative one.

## Start selection

The maximum valid start reserves room for the target horizon:

```text
max_start = frame_count - 1 - k - (T-1)*s
```

- Training chooses a deterministic pseudo-random integer in `[0,max_start]`.
- Validation chooses the midpoint of that interval.
- If the video is too short, start is zero and out-of-range requests are clamped to the final frame,
  causing repeated terminal frames.

The clamp keeps the sample shape valid. It does not invent motion.

## Per-sample deterministic RNG

Each item derives its augmentation/start RNG from:

```text
SHA256(f"{base_seed}:{epoch}:{sample_id}:{TRANSFORM_VERSION}")
```

The first eight digest bytes are interpreted as an unsigned big-endian integer. Consequences:

- worker scheduling does not choose a different crop;
- retrying the same sample in the same epoch reproduces its transformations;
- changing the epoch intentionally changes training views;
- validation remains stable because its epoch/start policy is stable;
- a transformation-version change causes a new seed stream.

This is separate from the global Torch RNG used for flow noise and model stochasticity.

## Decode path

1. Decord opens the video with one decoding thread.
2. The dataset forms the union of required context and target indices.
3. It decodes only those indices, rather than the entire video.
4. Duplicate requests are allowed; short videos may repeat the last frame.
5. Decord RGB `uint8` is converted to floating point and divided by 255.

The decode-only-needed-frames optimization historically reduced step time from roughly 1.66 to
1.41 seconds in an early setup. That is an experiment-history observation, not a current universal
throughput guarantee.

## Spatial and color transforms

The raw-data transform is:

1. resize the shorter side to 256 using bilinear interpolation with `align_corners=False`;
2. training: random 256×256 crop;
3. validation: center 256×256 crop;
4. training only: independent brightness, contrast, and saturation factors sampled in `[0.6,1.4]`;
5. clamp to `[0,1]`.

The same combined context+target tensor receives one crop and one set of color factors, preserving
pair consistency. There is no horizontal flip, rotation, hue jitter, or encoder normalization at
this layer. Encoder-specific normalization happens inside the encoder adapter.

## DataLoader order and exact resume

Training order is produced by a generator seeded with:

```text
base_seed + epoch * 1,000,003
```

Validation paths are sorted. Its loader seed uses `base_seed + 10,000`, but deterministic sorted
ordering is the important observable contract.

Training uses `drop_last=True`. Resume reconstructs the epoch order and starts at:

```text
epoch        = global_step // batches_per_epoch
batch_index  = global_step % batches_per_epoch
start_offset = batch_index * batch_size
```

The checkpoint also stores explicit sampler state. A resumed step must see the same next batch, not
merely a batch drawn from the same distribution.

## Tiny SSv2 arithmetic

The deterministic tiny subset contains 174 classes:

- training: 23 videos/class = `174 × 23 = 4,002`;
- validation: 2 videos/class = `174 × 2 = 348`;
- selection seed: 42.

At batch size 64:

```text
floor(4002 / 64) = 62 batches/epoch
62 * 64 = 3968 consumed samples/epoch
4002 - 3968 = 34 dropped tail samples/epoch
```

A 15,000-step run processes `15,000 × 64 = 960,000` example presentations. In loader epochs:

```text
15,000 = 241 * 62 + 58
```

So it completes 241 full loader epochs and 58 batches of the next. It is emphatically not “one
epoch.”

## Full SSv2 arithmetic

The repository records:

| Split | Videos |
|---|---:|
| train | 168,913 |
| validation | 24,777 |

At batch size 64:

```text
floor(168913 / 64) = 2639 batches/epoch
2639 * 64 = 168896 consumed
17 dropped
15000 / 2639 ≈ 5.684 epochs
```

Any prose calling 15,000 steps one full-SSv2 epoch is stale.

## EGO chunk semantics

EGO clips are fixed four-second chunks derived from longer source videos. “Video count” can mean
either source recordings or model-ready chunks, so use the precise noun:

- `source_uid`: original source recording;
- `chunk`: one four-second encoded training item;
- `split`: source-level assignment, preventing source leakage;
- `manifest row`: authoritative record of one chunk.

The tiny EGO subset has exactly 4,000 training chunks and 350 validation chunks, capped at ten chunks
per source. The full builder targets roughly 210 source-hours before chunking; exact output counts
must be read from the produced manifest, not inferred as a permanent constant.

## Memory arithmetic for raw batches

One FP32 clip contains:

```text
8 * 3 * 256 * 256 * 4 bytes = 6 MiB
```

At batch 64:

- one clip tensor: 384 MiB;
- context plus target: 768 MiB.

This excludes temporary transforms, normalized copies, detailed features, activations, gradients,
optimizer state, CUDA allocator overhead, and the frozen encoder itself. It is a lower-bound tensor
ledger, not a GPU-memory forecast.

## Failure and leakage checklist

- Context/target overlap is expected at `k=4`; do not describe it as disjoint prediction.
- Never use chunk-level random splitting for EGO; source-level leakage would invalidate validation.
- A short-video clamp can create static repeated frames; inspect manifests and duration filters.
- Shared augmentations protect temporal consistency; applying independent jitter to each window
  would create a synthetic prediction discrepancy.
- `drop_last` changes exact example counts and resume math.
- A dataset path alone is not dataset identity; inventories, frame counts, manifests, and hashes
  matter.
- “Frame” may mean decoded RGB frame, temporal encoder plane, or token. State which one.
