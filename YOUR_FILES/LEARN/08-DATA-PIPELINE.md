# 08 — The Data Pipeline: Video Decoding, Dataloaders, and the CPU Bottleneck

> **What you'll understand after this file:** the journey of one training
> sample from `.webm` file to GPU tensor, why video data pipelines are
> CPU-bound, what we optimized and what remains, the augmentation policy
> and its constraints, and the throughput/cost arithmetic of a run.

---

## 1. Why video data is an MLOps problem in itself

An image dataset is decoded once into tensors and you're done. Video is
different in kind:

- **Videos are compressed streams**, not arrays. SSv2 ships as `.webm`
  files encoded with **VP9** — a codec that stores most frames as *deltas*
  against previous frames (only occasional "keyframes" are complete
  images). Reading frame 40 may require decoding frames 32–40.
- **Decoding is CPU work.** Each frame must be decompressed before it's a
  pixel array. For our pipeline this is the single largest CPU cost.
- **We use a tiny window of each video.** Each sample needs 16 specific
  frames (8 context + 8 target) out of a clip's ~30–90 — so a naive
  "decode everything" wastes 2–5× the decode work.

Meanwhile the GPU consumes batches at its own rate. The design goal of any
data pipeline: **the GPU must never wait.** Every second the GPU idles is
rented compute burned on nothing.

## 2. The journey of one sample (`SSV2Dataset.__getitem__`)

```
.webm file
  → _open_video_reader: decord VideoReader, num_threads=1  (no decoding yet)
  → len(reader): frame count, cheap (container metadata)
  → _window_indices: pick the 8+8 frame indices (random start for train)
  → _decode_frames: reader.get_batch(16 indices)            ← THE expensive line
  → uint8 (16,H,W,3) → float [0,1], permute to (16,3,H,W)
  → _resize_shorter_side → 256
  → _crop: ONE random 256×256 crop for all 16 frames (train) / center (val)
  → _color_jitter: ONE sampled brightness/contrast/saturation for all 16
  → _normalize_encoder: ImageNet mean/std (the encoder's contract, file 02)
  → split: first 8 = context_clip, last 8 = target_clip
```

Three design decisions worth understanding deeply:

**(a) Decode-only-what-you-need.** The original code decoded *every* frame
then indexed. The optimization pass split reading into "open reader, get
length" and "decode exactly these 16 indices" via decord's random-access
`get_batch`. Why it helped less than hoped (~15%, not 2–5×): VP9's
delta encoding means random access still decodes from the nearest keyframe
forward — you pay a *seek cost* per requested index. The lesson
generalizes: **I/O optimizations are bounded by the storage format.** The
real 2–5× lives in re-encoding the dataset (all-keyframe / raw frames) or
caching decoded tensors — heavier moves, deliberately deferred.

**(b) `num_threads=1` — the counterintuitive one.** decord's threaded
FFmpeg decoder hits a known race on some VP9 packets (EAGAIN, "Error
sending packet"; dmlc/decord #83/#145/#246) when threads > 1. So each
*reader* is single-threaded, and parallelism comes from a different layer:
8 DataLoader workers each decoding *different videos* concurrently.
Parallelize across items, not within items — same total throughput, none
of the codec's thread bugs. (General pattern: when a library's internal
parallelism is buggy, move the parallelism up one level.)

**(c) Shared augmentation across the 16 frames.** The crop window and the
jitter parameters are sampled *once* per sample and applied to context and
target alike. Mandatory, not stylistic: if the context were cropped from
the top-left and the target from the bottom-right, "predict the future"
would silently become "predict the future *somewhere else in the frame*" —
an impossible task injected by the pipeline. Augmentations must never
change the *relationship* the model is supposed to learn; they may only
change nuisance factors (absolute position, color balance). This is also
why there are **no horizontal flips**: SSv2 labels are direction-sensitive
("pushing left to right"), and flipping reverses the semantics of motion.

## 3. The DataLoader layer

```python
num_workers = 8          # parallel decoding processes
pin_memory = True
shuffle = True (train)   # decorrelates batches
drop_last = True (train) # constant batch size — variance floor & batch
                         #   statistics behave consistently
```

How it works: 8 worker *processes* (not threads — this sidesteps Python's
GIL for CPU-bound decode work) each execute `__getitem__` independently
and feed a queue; the main process collates 64 samples into the batch
tensors. `pin_memory` puts batches in page-locked RAM so the CPU→GPU copy
can be async DMA (`non_blocking=True` on the `.to(device)` in
`train_step` completes the pattern).

The capacity question you should be able to answer for any pipeline:
*workers × (samples per worker-second) vs GPU's (samples per second).* If
decode-per-sample ≈ 0.5s, 8 workers deliver ~16 samples/s ≈ one batch of
64 every 4s. If the GPU's forward+backward takes less than that, the GPU
starves — and that, not FLOPs, sets your training speed. This is precisely
our regime: Run 1 averaged ~1.4 s/step where the pure GPU compute is a
fraction of that. **We are dataloader-bound**, which is why the
optimization Kanban (01-OPTIMIZE-DATALOADER) targets CPU work, and why a
"bigger GPU" would not make this training faster — but a higher-CPU-count
pod would.

Diagnosing which side you're on, in general: GPU utilization sawtoothing
(spikes then idle) → input-bound; pegged at ~100% → compute-bound. (Watch
`nvidia-smi` / pod metrics during the first minutes of any run.)

## 4. What we deliberately did NOT do (and the trade-offs)

These came up in the tech lead's "shift work to GPU / parallelize" review;
each is real but was deferred with reasons — know them so you can re-open
the decisions when conditions change:

| Optimization | Win | Why deferred |
|---|---|---|
| **Pre-encode e_t to disk** (encoder is frozen — features never change!) | eliminates 2 encoder forwards/sample AND most decode work | augmentations happen in *pixel* space before encoding; caching freezes the crop/jitter, shrinking effective data diversity — risky precisely while we're worried about effective rank. Cacheable variant: store N augmented encodings per clip (storage blow-up). |
| **GPU video decoding** (NVDEC via decord/DALI) | moves decode off CPU entirely | new dependency surface, VP9-on-NVDEC support is uneven, competes with training for GPU; wrong move while a simpler CPU fix exists |
| **Re-encode dataset** (keyframe-only / JPEG frames) | kills the seek cost — the actual bottleneck | hours of one-time CPU + ~10× storage for frames; sensible before *full-SSv2* runs, overkill for tiny |
| **Bigger num_workers** | linear-ish until CPU count saturates | pod had limited cores; workers beyond physical cores thrash |

The meta-lesson: data pipeline optimization is a *ladder* — measure, take
the cheapest rung that unblocks you, re-measure. We took "decode 16 frames
instead of all" (one function refactor), measured ~15%, identified the
seek cost as the next wall, and consciously stopped before the expensive
rungs, because the run we needed to launch was viable at ~1.4 s/step.

## 5. The arithmetic you should be able to do on a napkin

Throughput and cost for Run-style planning (numbers from Run 1):

```
s/step ≈ 1.4            (measured, dataloader-bound)
15,000 steps × 1.4 s    ≈ 21,000 s ≈ 5.8 h wall-clock
RunPod A100 @ ~$1.6/h   ≈ $9–10 per full run
samples seen            = 15,000 × 64 = 960k clip-pairs (240 epochs of tiny)
```

Every planning argument in the Kanban — why 30k was cut to 15k, why
checkpoints every 2.5k, what a crash at step 10.7k cost — is this
arithmetic. Habit to build: before launching any run, write down s/step ×
steps = hours × $/h = cost, and steps ÷ steps-per-epoch = epochs. Two
lines, and you'll never be surprised by a 5-hour job or a 480-epoch
overtrain again.

## 6. Datasets and splits

- **Full SSv2:** ~220k videos total (~170k train), ~24 GB of .webm. The
  intended Phase 1 corpus in the original spec.
- **ssv2_tiny:** ~4k-clip symlink subset used for fast iteration. Symlinks
  (not copies) — the subset is a *view* into the full dataset, so it costs
  no storage and is regenerable by seed.
- **Validation split:** used for the fixed diagnostic batch (file 06) —
  center-crop, no jitter, deterministic windows, so diagnostics measure
  the model, not the augmentation dice.

The path contract: everything lives under `/workspace/data/<dataset>` on
the pod (`JEPA_DATA_ROOT` overrides for local work). `/workspace` is the
RunPod *persistent volume* — data and checkpoints survive pod restarts;
everything else (pip packages! tmux!) does not. That asymmetry is the
root cause of the "fresh pod has no transformers" incident — see file 09.

## 7. Questions to test yourself

1. Why doesn't `get_batch([10, 20, 30])` cost 3 frame-decodes? *(VP9 delta
   encoding: each random access decodes from the nearest keyframe forward;
   you pay seek costs.)*
2. Why is parallelism at the DataLoader-worker level instead of decoder
   threads? *(decord's threaded VP9 decode is buggy (EAGAIN); workers
   parallelize across videos — same throughput, no codec race.)*
3. Why must context and target share one crop? *(Different crops change
   the relationship being learned — the model would have to predict the
   future at a different location; the task must stay "same view, later
   time.")*
4. Your GPU shows 35% utilization sawtoothing during training. Diagnosis
   and two fixes? *(Input-bound: GPU waits on dataloader. Fixes: more
   workers/CPU cores, cheaper decode (re-encode dataset, cache decoded
   frames or features).)*
5. Why no horizontal flip augmentation on SSv2? *(Direction-sensitive
   actions; flips reverse motion semantics.)*
