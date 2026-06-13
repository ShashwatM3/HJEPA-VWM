# 03 — CPU-to-GPU offload (contingency)

> **This folder is CONDITIONAL.** Do not execute unless Plan Phase 01 leaves
> `s/step > 0.7` after the dataloader change *and* the pod's vCPU count is
> already in the healthy range (Task 1B.1). If both are true and throughput is
> still poor, the remaining slack is CPU-bound augmentation and (worst case)
> decode itself. Those move to GPU.

## What this task is

Up to three increasingly invasive changes to `data.py` and the train step, each
of which moves work currently done on CPU workers onto the GPU. Done in order;
**stop as soon as `s/step` is acceptable**.

| Sub-phase | What moves to GPU | Effort | Expected gain |
|---|---|---|---|
| **3A** | float cast + normalize + color jitter | small | 10–20% |
| **3B** | resize + crop | medium | 5–15% on top of 3A |
| **3C** | VP9 decode itself (NVDEC via `torchcodec`) | large | up to ~3× on top of 3A+3B |

## Why CPU-to-GPU works here

Two reasons.

1. **The GPU is currently idle waiting for CPU**, per the measured
   `(user+sys) / real ≈ 7.8` ratio on the smoke run. Anything we add to the GPU
   pipeline is essentially free until GPU utilization catches up to ~100%.
2. **The ops we're moving are GPU-trivial.** A tensor `.float() / 255.0`, a
   bilinear resize, or a per-channel normalize on a `(64, 16, 3, 256, 256)`
   tensor is microseconds on an A100. The reason they cost meaningful CPU time
   today is that they run **per video, per CPU worker** rather than batched
   across the whole batch on the GPU.

## Sub-phase 3A — float / jitter / normalize on GPU

### What changes

In `data.py`, `__getitem__` currently returns `float32` frames in
`(8, 3, 256, 256)` shape after `_color_jitter` + `_normalize_encoder`. After 3A:

- `__getitem__` returns **`uint8`** frames in `(8, 3, 256, 256)` shape after
  resize + crop **only** (no float cast, no jitter, no normalize).
- The training loop, immediately after `.to(device)`, does the float cast
  (`.float() / 255.0`), then color jitter (vectorized across the batch), then
  normalization, all on GPU.

### Why this is a clean win

- `uint8` is **4× cheaper to transfer** from CPU pinned memory to GPU than
  `float32` (1 byte vs 4 bytes per element). With batch 64 × T=16 × 3 × 256 ×
  256, that's 50 MB vs 200 MB per batch — significant when copies happen every
  step.
- Per-batch GPU jitter sees all 64 videos at once and uses one cuDNN-fast set of
  broadcasted multiplies, vs CPU loops over individual videos.

### Risk

Low, but watch one thing: RNG order. Color jitter is per-batch-element today
(one set of `brightness/contrast/saturation` per video). If we batch on GPU we
must preserve "one set per video," not "one set per batch." Implementation: pass
the seeded random params as a `(B,)` tensor.

## Sub-phase 3B — resize + crop on GPU

### What changes

`_resize_shorter_side` and `_crop` currently run on CPU on per-video float
tensors. After 3B:

- `__getitem__` returns the **raw decoded** uint8 frames from decord,
  unresized and uncropped, padded to a common max size so the dataloader can
  collate into a tensor batch.
- The training loop resizes and crops on GPU after `.to(device)`.

### Complication: variable native resolution

SSv2 videos are roughly 240p but exact dimensions vary slightly. Three ways to
handle:

1. **Pre-resize on CPU to a common pre-crop size** (e.g. shortest side → 256),
   then crop on GPU. *Simplest; recommended.*
2. Pad raw frames to a single max H×W on CPU, then GPU resize + crop. *Most
   savings, more complex.*
3. Bucketed batching by native size. *Overkill for SSv2.*

Default to (1) unless empirically (1) doesn't move the needle.

### Risk

Medium. Bilinear resize on GPU vs CPU can produce slightly different float
values at edge pixels — not numerically identical to the pre-3B run, but
model-quality-equivalent. Document this in the commit message: "after 3B,
results are no longer bit-reproducible against pre-3B runs."

## Sub-phase 3C — GPU VP9 decode via `torchcodec`

### What changes

Replace `decord` in `data.py` with
[`torchcodec.decoders.VideoDecoder`](https://github.com/pytorch/torchcodec),
which uses NVIDIA's NVDEC hardware block to decode VP9 directly into GPU memory.
The CPU workers no longer touch raw video bytes after the initial file read.

### Why this could be the biggest win

VP9 decode is currently the dominant CPU cost per video. NVDEC on A100 supports
VP9 in hardware (separate silicon from CUDA cores, so it does not slow the
model's forward pass meaningfully). Frames are decoded directly into device
memory — no CPU→GPU copy of pixel data.

This is also the path the **official Hugging Face V-JEPA 2 docs** use in their
example:

```python
from torchcodec.decoders import VideoDecoder
import numpy as np

vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64)
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
```

(Source: [transformers v4.56.2 V-JEPA 2 doc](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/vjepa2))

So we'd actually be aligning the data path with the encoder's upstream reference.

### Risk

Higher than 3A/3B for three reasons:

1. `torchcodec` is newer than `decord`; rough edges possible. Mitigation: keep
   `decord` as a fallback at first and gate the switch behind a config flag.
2. NVDEC is shared with the GPU. While it does not contend for CUDA cores, it
   does compete for memory bandwidth. Net win is still large but not as large as
   the raw NVDEC throughput numbers might suggest.
3. Install: `torchcodec` needs to be matched against the pod's CUDA version and
   FFmpeg build. RunPod's PyTorch templates vary; verify the install before
   refactoring.

### When NOT to do 3C

- If `s/step ≤ 0.5` after 3A and 3B — the marginal pod-cost saving (a few
  dollars per Phase 1 run) does not justify the refactor risk.
- Before all of 3A and 3B are done and measured — you'd be optimizing without
  knowing whether they alone were sufficient.

## Success criteria / goals

Per sub-phase, the goal is to lower `s/step` and confirm model quality is
unchanged.

| Sub-phase | Target s/step | Quality check |
|---|---|---|
| After 3A | ≤ 0.6 | step-0 loss within ~5% of pre-3A on the same seed |
| After 3B | ≤ 0.5 | step-0 loss within ~10% (geometric ops differ pixel-level on GPU) |
| After 3C | ≤ 0.4 | bypass-test ratios at step ~5k match the decord baseline within noise |

If you hit the target s/step at any sub-phase, **stop optimizing and proceed to
Plan Phase 02**.

## How this fits the bigger picture

This is purely a throughput optimization. It does not change:

- The architecture (`E`, `B`, `B_EMA`, `F_c` are untouched).
- The loss (`L_flow + 0.10 · L_var`).
- The schedule (warmup, EMA, LR, EMA momentum).
- The acceptance gates (PHASE_1.md §12 still apply).
- The frame indices the model sees (same `_window_indices` logic).

So the question is purely: is the marginal pod-cost saving worth the refactor
effort? Answer: only if 01 alone isn't enough.

## What this task is NOT

- **NOT** a model change. If you find yourself editing `models.py` or
  `losses.py`, stop — you've drifted scope.
- **NOT** a dependency rebellion. Do not add `nvidia-dali`, ffmpeg python
  wrappers, or other heavyweight video libraries without escalating.
- **NOT** an excuse to rewrite the training loop. The data path is what's
  moving; everything else stays.
