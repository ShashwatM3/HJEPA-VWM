# 01 — Optimize the dataloader

## What this task is

Two changes that together reduce CPU-side data preparation time:

1. **Code change to `data.py`** — decode only the ~16 video frames the model
   actually consumes per step, instead of decoding **all** frames in each video
   and then slicing them down.
2. **Pod-side health check** — confirm the pod has enough vCPUs to keep the 8
   dataloader workers parallel; flag for an upgrade if it doesn't.

## Why this is being done

### Measured baseline

A timed 200-step smoke run (commit `4c1abb3`, `--data ssv2_tiny --steps 200`)
took **5m31s wall-clock** and consumed **~43 minutes of CPU time** across cores.
That gives:

- **`s/step ≈ 1.66`** at 64-video batches.
- Extrapolated to the canonical 30k Phase 1 run: **~13.8 hours of pod time**.
- The `(user + sys) / real ≈ 7.8` ratio shows the 8 dataloader workers were
  CPU-saturated while the A100 idled waiting for batches. The bottleneck is
  data prep, not GPU compute.

### Root cause inside `data.py`

The current helper decodes every frame of a video before `__getitem__` discards
~75% of them:

```python
# data.py — current
def _read_video_decord(path: Path) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return reader.get_batch(list(range(len(reader)))).asnumpy()
```

```python
# data.py — __getitem__
frames_np = _read_video_decord(self.paths[index])
context_idx, target_idx = self._window_indices(len(frames_np))
...
stacked = torch.from_numpy(frames_np[context_idx + target_idx])
```

We use `cfg.model.t_ctx = 8` context frames + 8 target frames = **16 frames per
video**. SSv2 clips are commonly 30–90 frames. So per video we decode ~3–6× more
work than we consume.

The fix is the documented decord pattern: open the reader (metadata only),
compute the 16 indices we need, then `reader.get_batch(indices)` to decode only
those frames. The frames returned are **bit-identical** to the ones we currently
keep — same indices, same stride, same downstream resize/crop/normalize. There is
zero impact on what the model sees.

### Why `num_threads=1` stays

The decord `num_threads=1` constraint added in commit `10adaa4` is required —
SSv2 ships as VP9-encoded `.webm` and decord's threaded FFmpeg decoder fails with
EAGAIN (-11) on some VP9 packets when threads > 1
([dmlc/decord#83](https://github.com/dmlc/decord/issues/83),
[#145](https://github.com/dmlc/decord/issues/145),
[#246](https://github.com/dmlc/decord/issues/246)). Per-worker parallelism is
gone; only across-worker parallelism remains. That makes the
"decode-only-needed-frames" optimization even more leveraged — it directly cuts
the serial work each worker does per video.

## Success criteria / goals

| Goal | Target |
|---|---|
| s/step after 1A (`time` 200-step run) | ≤ 0.8 (target), ideally 0.5–0.7 |
| Extrapolated 30k Phase 1 wall-clock | ≤ 7 hours (target), ideally 5 |
| Model loss values vs baseline at step 0 | ≤ 5% relative deviation (sampling noise only) |
| Stage 0 sanity (`--stage0-only`) | Still passes; encoder still frozen |
| vCPU count on the pod | Confirmed ≥ 12; flagged if < 12 |
| Code quality | Conforms to [`../../AGENT-BEHAVIOUR/CODE_DESIGN.md`](../../AGENT-BEHAVIOUR/CODE_DESIGN.md) (docstring with shape contract, no scattered detaches, etc.) |

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `get_batch(indices)` returns frames in a different order than `frames_np[indices]` did | Very low | decord docs guarantee returned order matches `indices`; verify with stage0 sanity (synthetic) + smoke (real) |
| VP9 keyframe spacing forces decord to decode many "in-between" frames anyway, giving less speedup than expected | Medium | Measure s/step empirically before committing to further optimization; if savings are < 2×, escalate to Plan Phase 3 |
| `len(reader)` blocks on something heavier than metadata-only read | Very low (decord docs explicit) | Same — measure |
| Pod's pyexpat/torch install gets confused by the change | None — we don't change any deps | N/A |

## How this fits the bigger picture

| | Before 01 | After 01 (target) |
|---|---|---|
| Full 30k Phase 1 wall-clock | ~14 h | ~5–7 h |
| Pod cost per full run ($2–3/hr A100) | $28–42 | $10–18 |
| Phase 2 / 3 runs (same data path) | inherit slow path | inherit fast path |
| Bit-identical model behaviour | n/a | yes |

The optimization is paid once and compounds across every subsequent training run
(Phase 2 fine flow, Phase 3 frame generator, any re-runs on the full SSv2). It is
the highest-leverage change in the entire plan.

## What this task is NOT

- **NOT** a model or loss change. Architecture is untouched.
- **NOT** a deviation from [`../../PHASES/PHASE_1.md`](../../PHASES/PHASE_1.md)
  acceptance gates. Those still hold.
- **NOT** the time to evaluate GPU-side decoding (NVDEC). That is folder
  [`../03-CPU-TO-GPU-OFFLOAD/`](../03-CPU-TO-GPU-OFFLOAD/), conditional on this
  folder not being enough.
