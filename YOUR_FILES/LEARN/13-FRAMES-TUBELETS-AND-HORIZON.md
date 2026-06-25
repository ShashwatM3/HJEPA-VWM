# Frames, Tubelets, And Horizon

> **What you'll understand after this file:** exactly how raw video frames
> become context/target clips, how tubelets relate to frames, and why
> `horizon_k=12` (not 4) is the current operating default.

---

## Current Pipeline (`horizon_k = 12`)

**1. Pick one video.**

We choose a starting point, call it frame `0`.

**2. Build the context clip.**

We take 8 sampled frames, every 2 frames:

```text
context frames = 0, 2, 4, 6, 8, 10, 12, 14
```

So the context clip is:

```text
8 sampled frames = 4 V-JEPA tubelets
```

because V-JEPA groups every 2 frames into 1 tubelet.

**3. Build the target clip.**

Current operating `horizon_k = 12`, so the target starts 12 frames later:

```text
target frames = 12, 14, 16, 18, 20, 22, 24, 26
```

The target clip is also:

```text
8 sampled frames = 4 V-JEPA tubelets
```

**Overlap note:** context ends at frame 14; target starts at frame 12.
Only frames **12 and 14** appear in both clips — minimal overlap. This is
deliberate: at the old `horizon_k=4`, six of eight target frames overlapped
the context, making "copy-ish" prediction too easy and letting low-rank
latents survive (investigation 003).

**4. V-JEPA encodes the context.**

Input:

```text
8 context frames
```

V-JEPA turns that into:

```text
4 temporal tubelet groups × 16 × 16 spatial patches = 1024 tokens
```

So:

```text
context frames -> context tubelet tokens e_t
```

**5. Bottleneck compresses context tokens.**

Input:

```text
1024 context tubelet tokens
```

Output:

```text
c_t = 32 abstract tokens
```

This is the model's compressed understanding of the context clip.

**6. V-JEPA encodes the target.**

Input:

```text
8 target frames
```

V-JEPA turns it into:

```text
1024 target tubelet tokens
```

So:

```text
target frames -> target tubelet tokens e_plus
```

**7. EMA bottleneck compresses target tokens.**

Input:

```text
1024 target tubelet tokens
```

Output:

```text
c_plus = 32 target abstract tokens
```

This is the thing we want the model to predict.

**8. Flow model learns prediction.**

Input condition:

```text
c_t = abstract latent of context clip
```

Target:

```text
c_plus = abstract latent of target clip
```

So the model learns:

```text
given the context clip's 4 tubelets,
predict the target clip's 4 tubelets
```

More precisely:

```text
given 4 context tubelets starting at frame 0,
predict 4 target tubelets starting at frame 12
```

At `horizon_k=12` with `frame_stride=2`, the prediction horizon is
**12 raw frames ≈ 1.0 second** ahead (SSv2 is ~12 fps).

---

## Legacy Pipeline (`horizon_k = 4`) — config default only

`config.py` still defaults to `horizon_k=4` for backward compatibility with
the original v0.2 spec. Do not use this for new runs unless ablating.

Context stays the same:

```text
context frames = 0, 2, 4, 6, 8, 10, 12, 14
```

Target at k=4:

```text
target frames = 4, 6, 8, 10, 12, 14, 16, 18
```

```text
given 4 context tubelets starting at frame 0,
predict 4 target tubelets starting at frame 4
```

Six of eight target frames overlap the context window. Copy baselines are
easier to beat superficially, but the task does not force the bottleneck to
use its full capacity — rank plateaued at ~5–9 under this setting.

---

## Quick comparison

| | `horizon_k=4` (legacy default) | `horizon_k=12` (operating) |
|---|---|---|
| Target start frame | 4 | 12 |
| Time ahead (12 fps) | ~0.33 s | ~1.0 s |
| Context/target overlap | 6 of 8 frames | 2 of 8 frames |
| Empirical rank | ~5–9 | ~13–14+ |
| CLI | (config default) | `--horizon-k 12` |

Everything else — tubelet geometry, token counts, bottleneck, EMA target,
flow matching — is identical. Horizon only changes *which* future window
the dataloader samples.
