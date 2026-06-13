# 03 — Tasks: CPU-to-GPU offload (coding agent)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — what this contingency
>   is and the per-sub-phase risk/effort table.
> - [`HUMAN_TASKS.md`](HUMAN_TASKS.md) — the human runs every re-timing
>   benchmark; you pause and wait between each sub-phase.
> - [`../README.md`](../README.md) — the agent/human handoff convention.
>
> **Conditional folder.** Do not execute unless:
>
> 1. Plan Phase 01 is closed AND
> 2. The post-01 `s/step` reported by the human is > 0.7 AND
> 3. The human has explicitly said "proceed to Plan Phase 03."
>
> Execute sub-phases in order **3A → 3B → 3C**. After each, pause for the
> human to re-time on the pod. **Stop optimizing the moment `s/step` is
> acceptable.**

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> Confirm the human has authorized Plan Phase 03 before starting Task A3.1.
> If not authorized, do not modify code.

---

# Sub-phase 3A — Float / jitter / normalize on GPU

## Task A3.1 — Modify `__getitem__` to return uint8 frames

**Status:** [NOT DONE]

**Description:**
Remove the per-video CPU-side float cast, color jitter, and normalize from
`__getitem__`. Return `uint8` frames after resize + crop only.

**Instructions:**

In [`../../../data.py`](../../../data.py), in `SSV2Dataset.__getitem__`:

1. **Remove** these lines:
   ```python
   if self.split == "train":
       stacked = _color_jitter(stacked)
   stacked = _normalize_encoder(stacked)
   ```
2. **Replace** the float cast line:
   ```python
   stacked = stacked.float().permute(0, 3, 1, 2) / 255.0
   ```
   with a permute-only version (still uint8):
   ```python
   stacked = stacked.permute(0, 3, 1, 2).contiguous()  # uint8 (T, C, H, W)
   ```
3. Update `_resize_shorter_side` to accept uint8 in and return uint8 out
   (the interpolation must still be float; cast in/out inside the function):
   ```python
   def _resize_shorter_side(frames: Tensor, size: int) -> Tensor:
       _require_torch()
       _, _, h, w = frames.shape
       scale = size / min(h, w)
       new_h = int(round(h * scale))
       new_w = int(round(w * scale))
       resized = F.interpolate(
           frames.float(), size=(new_h, new_w),
           mode="bilinear", align_corners=False,
       )
       return resized.round().clamp_(0, 255).to(torch.uint8)
   ```
4. Update the docstring for `__getitem__` to say it returns uint8 frames;
   the caller (training/diagnostics) is responsible for float cast + jitter +
   normalize.
5. Update `smoke_test_dataloader` assertions: replace
   `context_clip.min() < -1.0 or context_clip.max() > 1.0` with
   `context_clip.dtype == torch.uint8 and 0 <= context_clip.min() <= context_clip.max() <= 255`.

**How to verify:**

- `python -c "import ast; ast.parse(open('data.py').read())"` passes.
- The new docstring follows
  [`../../AGENT-BEHAVIOUR/CODE_DESIGN.md`](../../AGENT-BEHAVIOUR/CODE_DESIGN.md)
  §4.

---

## Task A3.2 — Move float / normalize / jitter to GPU in `train.py`

**Status:** [NOT DONE]

**Description:**
Add a small GPU-side preprocessing function that does what `__getitem__` no
longer does. Call it inside `_coarse_forward` (or just before) after `.to(device)`.

**Instructions:**

In [`../../../train.py`](../../../train.py), add a helper near the top (after
the imports, before `device_for_training`):

```python
_ENCODER_MEAN_GPU: Tensor | None = None
_ENCODER_STD_GPU: Tensor | None = None

def _gpu_preprocess(
    clip_u8: Tensor,
    device: torch.device,
    *,
    do_jitter: bool,
) -> Tensor:
    """Convert uint8 frames to encoder-normalised float on GPU.

    Args:
        clip_u8: (B, T, 3, H, W) uint8 tensor on `device`.
        device: target CUDA/CPU device.
        do_jitter: apply per-clip color jitter (training only).
    Returns:
        normalized: (B, T, 3, H, W) float tensor in the V-JEPA 2 expected range.
    """
    from config import ENCODER_IMAGE_MEAN, ENCODER_IMAGE_STD

    global _ENCODER_MEAN_GPU, _ENCODER_STD_GPU
    if _ENCODER_MEAN_GPU is None or _ENCODER_MEAN_GPU.device != device:
        _ENCODER_MEAN_GPU = torch.tensor(
            ENCODER_IMAGE_MEAN, device=device
        ).view(1, 1, 3, 1, 1)
        _ENCODER_STD_GPU = torch.tensor(
            ENCODER_IMAGE_STD, device=device
        ).view(1, 1, 3, 1, 1)

    x = clip_u8.float().div_(255.0)
    if do_jitter:
        b = x.shape[0]
        brightness = 1.0 + (torch.rand(b, device=device) * 0.8 - 0.4)
        contrast = 1.0 + (torch.rand(b, device=device) * 0.8 - 0.4)
        saturation = 1.0 + (torch.rand(b, device=device) * 0.8 - 0.4)
        x = x * brightness.view(b, 1, 1, 1, 1)
        mean_spatial = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean_spatial) * contrast.view(b, 1, 1, 1, 1) + mean_spatial
        gray = x.mean(dim=2, keepdim=True)
        x = (x - gray) * saturation.view(b, 1, 1, 1, 1) + gray
        x.clamp_(0.0, 1.0)
    return (x - _ENCODER_MEAN_GPU) / _ENCODER_STD_GPU
```

Then in `_coarse_forward` (or `train_step` before `_coarse_forward`),
preprocess both clips after `.to(device)`:

```python
context_clip = _gpu_preprocess(context_clip, device, do_jitter=True)
target_clip = _gpu_preprocess(target_clip, device, do_jitter=True)
```

> **Important:** the original CPU `_color_jitter` applied **the same** jitter
> params to context and target windows so they stayed photometrically
> consistent. Preserve this: in `_gpu_preprocess`, when called from
> `_coarse_forward`, accept an optional pre-seeded jitter param tuple from
> the caller, so context and target see the same brightness/contrast/saturation
> per batch element. Simpler implementation: stack context + target on the
> time dimension, preprocess once with `do_jitter=True`, then split.

For the diagnostics path (`run_diagnostics`), pass `do_jitter=False` for both
clips.

**How to verify:**

- `python -c "import ast; ast.parse(open('train.py').read())"` passes.
- The new helper has the shape contract.

---

## Task A3.3 — Commit and push

**Status:** [NOT DONE]

**Description:**
Land sub-phase 3A as a single commit.

**Instructions:**

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
git status               # data.py + train.py
git diff data.py train.py
git add data.py train.py
git commit -m "Move float/jitter/normalize to GPU (3A)

Dataloader returns uint8 frames after resize+crop only. Train loop runs
float cast, color jitter, and encoder normalization on the GPU as a single
batched op. Reduces CPU/worker time and CPU->GPU transfer (uint8 = 1/4 the
bytes of float32). Same per-batch-element jitter randomization as before;
context and target use the same jitter params per video."
git push origin phase1-v0.2-frozen-encoder
```

**How to verify:**

- `git log -1 --oneline` shows the 3A commit.
- Push succeeded.

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> The next step is the human pulling on the pod, running Stage 0 sanity,
> and running a fresh `time` 200-step benchmark to measure 3A's effect.
> See [`HUMAN_TASKS.md`](HUMAN_TASKS.md) **H3.1**.
>
> Wait for the human to report back:
>
> 1. Stage 0 sanity pass/fail.
> 2. New `s/step` after 3A.
> 3. Step-0 metric values (to confirm no regression).

---

## Task A3.4 — Decide whether to proceed to 3B

**Status:** [NOT DONE]

**Description:**
Apply the decision rule from
[`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md).

**Instructions:**

- `s/step ≤ 0.6` → **skip 3B and 3C**. Mark Tasks A3.5 onward as
  `[CANCELLED]`. Go to A3.10 (close-out).
- `s/step > 0.6` → proceed to A3.5.

Tell the human which path you're taking.

**How to verify:**

- Decision communicated.

---

# Sub-phase 3B — Resize / crop on GPU

> Only execute if A3.4 routed here.

## Task A3.5 — Move resize/crop out of `__getitem__`

**Status:** [NOT DONE]

**Description:**
Remove `_resize_shorter_side` and `_crop` from the CPU dataloader path. Add
a cheap CPU pre-resize to a common pre-crop size so the batch collator has
homogeneous shapes; do the final resize and crop on GPU.

**Instructions:**

1. In [`../../../data.py`](../../../data.py), in `__getitem__`:
   - After `_decode_frames`, do `_resize_shorter_side(stacked, 272)` (slightly
     larger than the 256 final size — gives the GPU a margin to crop from).
   - Remove the call to `_crop`.
   - Return uint8 `(T, 3, 272, ~)` frames.
   - If widths differ across videos, pad to a common width in `__getitem__`
     so default `torch.utils.data.default_collate` works.
2. In [`../../../train.py`](../../../train.py) `_gpu_preprocess`, before the
   float cast:
   - GPU bilinear resize from (272, W) to (256, 256).
   - GPU center crop (eval) or random crop (train) — generate a random
     `(top, left)` offset per batch element.
3. Update `smoke_test_dataloader` assertions for the new pre-GPU shapes.

**How to verify:**

- `python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"`
  passes with the new shape assertion.
- Syntax OK.

---

## Task A3.6 — Commit and push (3B)

**Status:** [NOT DONE]

**Description:**
Land sub-phase 3B as a single commit.

**Instructions:**

```bash
git add data.py train.py
git commit -m "Move resize/crop to GPU (3B)

Dataloader pre-resizes to a common (272, W) on CPU, then batches. Final
resize to 256 + random/center crop happens on GPU as batched ops. Bilinear
resize numerically differs from the CPU pre-3B path; not bit-reproducible
against earlier runs (model-quality-equivalent)."
git push origin phase1-v0.2-frozen-encoder
```

**How to verify:**

- 3B commit on HEAD; push succeeded.

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> Wait for the human to pull, re-run Stage 0 sanity, and time 200 steps
> on the pod (see [`HUMAN_TASKS.md`](HUMAN_TASKS.md) **H3.2**). Need from
> them: Stage 0 pass/fail, new `s/step`, step-0 metric values.

---

## Task A3.7 — Decide whether to proceed to 3C

**Status:** [NOT DONE]

**Description:**
3C is a real refactor with a new dependency. Don't enter it without explicit
tech lead approval (relayed via the human).

**Instructions:**

- `s/step ≤ 0.5` → **skip 3C**. Mark Tasks A3.8 / A3.9 as `[CANCELLED]`. Go to
  A3.10.
- `s/step > 0.5` → ask the human to get tech lead approval for 3C. Do not
  proceed without it.

**How to verify:**

- Decision communicated.

---

# Sub-phase 3C — NVDEC decode via `torchcodec`

> Only execute if A3.7 routed here AND the human relayed tech lead approval.

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> Before touching code, ask the human to run
> [`HUMAN_TASKS.md`](HUMAN_TASKS.md) **H3.3** — the torchcodec install +
> NVDEC sanity script. If CUDA decode fails on the pod's CUDA/FFmpeg
> combo, **abort 3C** (do not refactor `data.py`). Wait for the human to
> report:
>
> 1. Whether `pip install torchcodec` succeeded on the pod.
> 2. Whether the CPU decode sanity passed.
> 3. Whether the CUDA decode sanity passed (this is the gate — if no,
>    abort 3C).

---

## Task A3.8 — Refactor `data.py` to use `torchcodec` with decord fallback

**Status:** [NOT DONE]

**Description:**
Replace the decord-based `_open_video_reader` + `_decode_frames` with a
torchcodec equivalent. Keep the decord path behind a config flag for
fallback.

**Instructions (high-level — the exact diff depends on torchcodec's pinned
version):**

1. Pin a torchcodec version (record in `requirements.txt`). Read its docs
   for that version before writing code — the API has been in flux.
2. Add a config field in [`../../../config.py`](../../../config.py):
   ```python
   video_decoder: str = "torchcodec_cuda"  # "decord" | "torchcodec_cpu" | "torchcodec_cuda"
   ```
3. In [`../../../data.py`](../../../data.py), refactor the decode helpers to
   switch on `cfg.data.video_decoder`. The torchcodec path opens the decoder
   per video per worker; the CUDA variant decodes directly into GPU memory
   and returns a tensor already on `cuda:0`.
4. The dataloader path must not pin frames in CPU memory if they are already
   on GPU. Two options:
   - Have the dataloader return GPU tensors and use `num_workers=0`
     (workers can't easily share CUDA tensors).
   - Move the torchcodec CUDA decode out of the dataloader and into the
     train loop itself.
   Pick the simpler option (likely the second) and document the choice in
   the function docstring with a citation to the torchcodec version used.

**How to verify:**

- Stage 0 still passes locally if you can run it (this exercises the model,
  not the new decoder — so the import path is the main thing being tested).
- Syntax OK in all touched files.

---

## Task A3.9 — Commit, push (3C); pause for the human to re-time

**Status:** [NOT DONE]

**Description:**
Land sub-phase 3C and ask the human to do the final timing measurement.

**Instructions:**

```bash
git add config.py data.py train.py requirements.txt
git commit -m "GPU decode via torchcodec NVDEC (3C)

Replaces decord (CPU VP9 decode) with torchcodec on CUDA. Frames decoded
directly into GPU memory; CPU workers no longer touch raw video bytes
after the initial file read. decord path retained as fallback behind
cfg.data.video_decoder='decord'. Pinned torchcodec==<version>."
git push origin phase1-v0.2-frozen-encoder
```

Then **⏸ PAUSE — BLOCKED ON HUMAN.** Wait for them to do
[`HUMAN_TASKS.md`](HUMAN_TASKS.md) **H3.4**: pull + `pip install -r
requirements.txt` + Stage 0 + timed 200-step benchmark. Need:

1. New `s/step` after 3C.
2. Step-0 metric values (`L_var` within ~10% of the decord baseline).
3. Any decoder errors observed.

**How to verify:**

- 3C commit on HEAD; push succeeded.

---

## Task A3.10 — Close out Plan Phase 03

**Status:** [NOT DONE]

**Description:**
Whichever sub-phase you stopped at, document the final `s/step` and hand off
to Plan Phase 02.

**Instructions:**

1. Mark all `[NOT DONE]` items as `[DONE]` or `[CANCELLED]` (with reason).
2. Verify all `[NOT DONE]` items in [`HUMAN_TASKS.md`](HUMAN_TASKS.md) are
   `[DONE]` or `[CANCELLED]`.
3. Tell the human: "Plan Phase 03 closed at sub-phase <3A|3B|3C>. Final
   s/step on `ssv2_tiny` is <value>. Proceed to
   [`../02-LAUNCH-FULL-PHASE-1-RUN/`](../02-LAUNCH-FULL-PHASE-1-RUN/)."

**How to verify:**

- All tasks in both files closed.
- Handoff message delivered.
