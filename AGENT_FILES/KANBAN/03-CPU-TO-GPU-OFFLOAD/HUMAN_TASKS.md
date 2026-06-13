# 03 — Tasks: CPU-to-GPU offload (human operator)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — what this contingency is.
> - [`TASKS.md`](TASKS.md) — the agent's tasks and where it pauses for you.
> - [`../README.md`](../README.md) — the agent/human handoff convention.
>
> **Conditional folder.** Only execute if Plan Phase 01 left `s/step > 0.7`.
> Otherwise this folder is skipped entirely.
>
> Your role: at each sub-phase end, pull the agent's commit on the pod, run
> Stage 0 sanity, and run a `time` 200-step benchmark. Report back the timing
> and step-0 metric values.

---

## Task H3.1 — Re-time after 3A (float/jitter/normalize moved to GPU)

**Status:** [NOT DONE]

**Corresponds to:** the first pause point in [`TASKS.md`](TASKS.md), after A3.3.

**Description:**
Pull the agent's 3A commit on the pod, confirm Stage 0 still passes, run a
fresh 200-step benchmark, and report the new `s/step` plus the step-0
metric values to the agent.

**Instructions:**

```bash
ssh runpod-jepa
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline                  # should show the 3A commit

python train.py --stage0-only         # must still pass

time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

**How to verify:**

- Stage 0 prints `Stage 0 sanity passed: {...}`.
- 200-step benchmark completes with no NaN.
- `time` block shows `real`, `user`, `sys`.

**Report back to the agent:**

- Commit hash from `git log -1 --oneline`.
- Stage 0 pass/fail.
- `real` time from the 200-step benchmark.
- Step-0 values of `L_flow` and `L_var` (compare to baseline
  `L_flow ≈ 2.87`, `L_var ≈ 0.43` — within ~5% is expected).
- `grad_has_nan` at step 0.

---

## Task H3.2 — Re-time after 3B (resize/crop moved to GPU)

**Status:** [NOT DONE]

**Corresponds to:** the second pause point in [`TASKS.md`](TASKS.md), after A3.6.

**Description:**
Same workflow as H3.1, after the agent pushes the 3B commit.

**Instructions:**

```bash
ssh runpod-jepa
cd /workspace/hierarchal-jepa-flow-world-model
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline                  # should show the 3B commit

python train.py --stage0-only

time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

**How to verify:**

- Same as H3.1 — Stage 0 passes; 200 steps complete; no NaN.
- Note: after 3B, step-0 metric values may drift by ~5–10% from the decord
  baseline because GPU bilinear resize is not bit-equivalent to CPU. This
  is expected.

**Report back to the agent:**

- Commit hash.
- Stage 0 pass/fail.
- `real` time.
- Step-0 `L_flow`, `L_var`.
- Whether the values are within ~10% of the 3A baseline (not the original
  baseline) — this catches regressions but allows expected pixel-level
  differences.

---

## Task H3.3 — Sanity-check torchcodec install on the pod

**Status:** [NOT DONE]

**Corresponds to:** the third pause point in [`TASKS.md`](TASKS.md), before A3.8.

**Description:**
Before the agent refactors `data.py` to use torchcodec, prove that
torchcodec installs and that CUDA VP9 decode works on the pod's CUDA/FFmpeg
combo. If CUDA decode fails, sub-phase 3C must be aborted (no point
refactoring if the underlying decode doesn't work).

**Instructions:**

```bash
ssh runpod-jepa
cd /workspace/hierarchal-jepa-flow-world-model
pip install torchcodec     # any version for sanity; agent will pin later

python - <<'PY'
from torchcodec.decoders import VideoDecoder
import glob, torch

video = glob.glob("/workspace/data/ssv2_tiny/train/*.webm")[0]
print("Test video:", video)

# CPU decode sanity:
vr_cpu = VideoDecoder(video)
out_cpu = vr_cpu.get_frames_at(indices=list(range(8)))
print("CPU OK:", out_cpu.data.shape, out_cpu.data.dtype)

# CUDA decode sanity:
vr_cuda = VideoDecoder(video, device="cuda:0")
out_cuda = vr_cuda.get_frames_at(indices=list(range(8)))
print("CUDA OK:", out_cuda.data.shape, out_cuda.data.dtype, out_cuda.data.device)
PY
```

**How to verify:**

- `pip install torchcodec` succeeds.
- The python script prints non-empty shapes for both CPU and CUDA paths.
- The CUDA tensor's `.device` is `cuda:0`.

If CUDA decode fails (most common reasons: torchcodec wheel doesn't match
pod CUDA version, NVDEC doesn't have VP9 support enabled, FFmpeg build
mismatch), capture the full error and report it. Do not try to debug
torchcodec on the pod — that's outside this folder's scope.

**Report back to the agent:**

- Whether `pip install torchcodec` succeeded.
- Whether CPU decode worked.
- Whether CUDA decode worked.
- If CUDA decode failed, the full error message.

---

## Task H3.4 — Re-time after 3C (torchcodec NVDEC)

**Status:** [NOT DONE]

**Corresponds to:** the fourth pause point in [`TASKS.md`](TASKS.md), after A3.9.

**Description:**
Final timing measurement after the torchcodec switch. Same workflow as the
previous re-time tasks, with an extra `pip install -r requirements.txt` step
since the agent will have updated the pinned dependency.

**Instructions:**

```bash
ssh runpod-jepa
cd /workspace/hierarchal-jepa-flow-world-model
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline                  # should show the 3C commit

pip install -r requirements.txt       # picks up the pinned torchcodec version

python train.py --stage0-only

time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

**How to verify:**

- `pip install` exits 0.
- Stage 0 passes.
- 200-step benchmark completes with no NaN.

**Report back to the agent:**

- Commit hash.
- Whether `pip install` succeeded.
- Stage 0 pass/fail.
- `real` time.
- Step-0 `L_flow`, `L_var` (within ~10% of 3B baseline).
- Whether any torchcodec error appeared during the run.
