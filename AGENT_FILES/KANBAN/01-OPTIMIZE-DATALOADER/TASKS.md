# 01 — Tasks: Optimize the dataloader (coding agent)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — why we are doing this.
> - [`HUMAN_TASKS.md`](HUMAN_TASKS.md) — what the human is doing in parallel,
>   especially the items the agent waits on.
> - [`../README.md`](../README.md) — the agent/human handoff convention.
>
> **Order:** do tasks in numerical order. Mark `[NOT DONE]` → `[DONE]` only
> when the **How to verify** check passes. Stop and pause at every
> `⏸ PAUSE — BLOCKED ON HUMAN` marker.

---

## Task A1.1 — Refactor `data.py` to decode only the needed frames

**Status:** [DONE] (commit `5e78caa`)

**Description:**
Replace the all-frames decode helper with a split: one helper opens the decord
`VideoReader` (metadata only, no decoding), and `__getitem__` computes the 16
context+target frame indices from `len(reader)`, then asks decord to decode
exactly those indices. Same downstream resize/crop/normalize, no other
behavior change.

**Instructions:**

1. Open [`../../../data.py`](../../../data.py).
2. **Replace** the function `_read_video_decord` with the following two
   helpers (preserve the existing module-level imports and `_require_torch`
   helper):

   ```python
   def _open_video_reader(path: Path):
       """Open an SSv2 .webm with decord at num_threads=1 (no decoding yet).

       `num_threads=1` is required: SSv2 ships as VP9-encoded .webm and decord's
       threaded FFmpeg decoder fails with EAGAIN (-11, "Error sending packet")
       on some VP9 packets when threads > 1. See dmlc/decord#83, #145, #246.
       Returning the open reader (rather than decoded frames) lets the caller
       read `len(reader)` cheaply and then decode only the indices it needs.
       """
       try:
           from decord import VideoReader, cpu
       except ModuleNotFoundError as exc:  # pragma: no cover - RunPod dependency.
           raise RuntimeError("decord is required to read SSv2 videos") from exc
       return VideoReader(str(path), ctx=cpu(0), num_threads=1)


   def _decode_frames(reader, indices: list[int]) -> np.ndarray:
       """Decode the requested frame indices from an open decord VideoReader.

       Decodes only the frames in `indices` (random-access via decord
       `get_batch`), rather than the whole video, so a 16-frame consumer does
       not pay for decoding all ~30-90 frames per SSv2 clip.

       Args:
           reader: an open decord VideoReader (see `_open_video_reader`).
           indices: 1-D list of frame indices to decode, in the order returned.
       Returns:
           frames: (len(indices), H, W, 3) uint8 numpy array in the same
               order as `indices`.
       """
       return reader.get_batch(indices).asnumpy()
   ```

3. **Update `SSV2Dataset.__getitem__`** — replace only the first few lines
   of the method body. Current code:

   ```python
   def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
       """Return one encoder-normalized (context_clip, target_clip) pair."""
       frames_np = _read_video_decord(self.paths[index])
       context_idx, target_idx = self._window_indices(len(frames_np))
       t_ctx = self.cfg.model.t_ctx
       stacked = torch.from_numpy(frames_np[context_idx + target_idx])
       ...
   ```

   Change to:

   ```python
   def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
       """Return one encoder-normalized (context_clip, target_clip) pair."""
       reader = _open_video_reader(self.paths[index])
       context_idx, target_idx = self._window_indices(len(reader))
       t_ctx = self.cfg.model.t_ctx
       frames_np = _decode_frames(reader, context_idx + target_idx)
       stacked = torch.from_numpy(frames_np)
       ...
   ```

   Leave all subsequent lines unchanged (`.float().permute(0, 3, 1, 2) /
   255.0`, `_resize_shorter_side`, `_crop`, `_color_jitter`,
   `_normalize_encoder`, `return stacked[:t_ctx], stacked[t_ctx:]`).

4. Confirm `_read_video_decord` is no longer referenced anywhere:
   ```bash
   rg "_read_video_decord" .
   ```
   Expected: no matches. If any remain, update those call sites.

**How to verify:**

- `python -c "import ast; ast.parse(open('data.py').read()); print('data.py: syntax OK')"`
  prints `data.py: syntax OK`.
- `rg "_read_video_decord" .` returns nothing.
- The two new helpers and the updated `__getitem__` docstring follow
  [`../../AGENT-BEHAVIOUR/CODE_DESIGN.md`](../../AGENT-BEHAVIOUR/CODE_DESIGN.md)
  §4 (one-line summary, plain-English why, `Args`/`Returns` with shapes).

---

## Task A1.2 — Local Stage 0 sanity (best-effort)

**Status:** [SKIPPED] — `transformers` is not installed on the laptop; Stage 0 is re-run on the pod in H1.1.

**Description:**
The data path change is small but touches code that runs in every training
step. Try Stage 0 locally — it uses synthetic tensors, not real video, so it
exercises only the model code path. This is a fast first check.

If the local environment cannot run Stage 0 (no GPU, no encoder cache,
transformers version mismatch on the laptop, etc.), **skip this task** —
treat as `[DONE]` with a note. The Stage 0 check is re-run on the pod by the
human in `HUMAN_TASKS.md` task **H1.1**.

**Instructions:**

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
python train.py --stage0-only
```

**How to verify:**

- Exits 0 with `Stage 0 sanity passed: {...}` printed, OR
- Skipped because the local env can't run it (note this in the commit
  message of A1.3).

---

## Task A1.3 — Commit and push

**Status:** [DONE] — pushed `5e78caa` to `origin/phase1-v0.2-frozen-encoder`.

**Description:**
Land the change as a single small commit on `phase1-v0.2-frozen-encoder` so
the pod can pull it.

**Instructions:**

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
git status                # only data.py should be modified
git diff data.py          # review the diff
git add data.py
git commit -m "Decode only needed frames in dataloader

Refactor _read_video_decord into _open_video_reader + _decode_frames.
__getitem__ now opens the decord VideoReader, reads len(reader) (metadata
only), computes the context+target indices, and decodes only those ~16
frames per video instead of the whole clip. num_threads=1 preserved for
VP9 reliability. Model inputs are bit-identical."
git push origin phase1-v0.2-frozen-encoder
```

**How to verify:**

- `git log -1 --oneline` shows the new commit at HEAD.
- `git status` reports a clean working tree.
- Push succeeded (the command exited 0 without an authentication error).

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> The next series of checks (Stage 0 on pod, timed 200-step benchmark, vCPU
> spec check) all require SSH access to the RunPod pod and are listed in
> [`HUMAN_TASKS.md`](HUMAN_TASKS.md) as **H1.1, H1.2, H1.3**.
>
> The agent must wait until the human reports back **all four** of the
> following values before proceeding to Task A1.4:
>
> 1. Stage 0 sanity pass/fail on pod (H1.1).
> 2. The `real` time (e.g. `5m31s`) from the timed 200-step benchmark (H1.2).
> 3. The step-0 and step-50 values of `L_flow` and `L_var` from the new run
>    (H1.2) — needed to confirm the optimization didn't change model inputs.
> 4. The pod's `nproc`, CPU model line, and RAM total (H1.3).
>
> Do not make assumptions about these numbers. Do not proceed without them.

---

## Task A1.4 — Interpret the timing result and write the throughput decision

**Status:** [DONE] — new s/step = 1.41 (~15% reduction from 1.66 baseline). Extrapolated 30k = ~12 hr (~$24–36). Step-0 metrics bit-identical to baseline (refactor is behavior-preserving). Bottleneck is now per-worker decord-seek on VP9, not workers-vs-cores. Recommended Plan Phase 02. nproc=128 means no hardware upgrade is meaningful.

**Description:**
Using the values the human reported back from H1.2, compute the new s/step
and decide whether Plan Phase 02 can proceed, or whether Plan Phase 03
(CPU→GPU offload) is needed first.

**Instructions:**

1. From H1.2's reported `real` time, compute:
   ```
   s/step  = real_time_seconds / 200
   extrapolated_30k_hours = (s/step * 30000) / 3600
   ```
2. Cross-check the human's reported step-0 `L_flow` and `L_var` against the
   pre-optimization baseline (`logs/1.json`, step 0):
   - Baseline `L_flow` ≈ 2.87, `L_var` ≈ 0.43.
   - New values must be within ~5% (sampling-noise level). If they differ by
     more than ~10%, the optimization is **NOT** behavior-preserving and the
     agent must investigate before continuing.
3. Apply the throughput decision tree from
   [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md):

   | New s/step | Decision |
   |---|---|
   | ≤ 0.7 | Plan Phase 01 is sufficient. Mark Plan Phase 03 tasks as `[CANCELLED]`. Proceed to A1.5. |
   | 0.7 – 1.0 | Modest win. Continue to A1.5; reconsider Plan Phase 03 only if the human also reports `nproc < 12` in H1.3. |
   | > 1.0 | Optimization underperformed. Continue to A1.5 with vCPU info, but flag that Plan Phase 03 will likely be needed. |

4. Write the decision into a short note (chat message back to the human, or
   add a brief comment to the most recent commit message). Include:
   - The new s/step.
   - The extrapolated 30k wall-clock in hours.
   - The estimated pod cost ($2–3/hr A100).
   - Which next folder to execute (Plan Phase 02 or Plan Phase 03).

**How to verify:**

- The decision note has been written and shared with the human.
- The chosen next folder is documented.

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> Task **H1.4** (pod-tier upgrade decision) requires the human to look at
> their RunPod billing situation and decide whether to redeploy at a higher
> CPU tier. The agent cannot make this call alone.
>
> Wait for the human to report back the final decision: either
>
> - "Keeping the current pod," or
> - "Redeploying — new pod is up at <commit hash> with <new nproc> vCPUs and
>   <new RAM> RAM."

---

## Task A1.5 — Close out Plan Phase 01

**Status:** [DONE] — all agent tasks DONE/SKIPPED; all human tasks DONE. Human picked Plan Phase 02 as next folder. Plan Phase 03 deferred (not cancelled — kept available for post-Phase-1 throughput work if needed).

**Description:**
Once the human has confirmed the final pod state and the s/step measurement
is in, declare Plan Phase 01 complete and hand off to Plan Phase 02 or 03 per
the decision in A1.4.

**Instructions:**

1. Mark all `[NOT DONE]` items in this file as `[DONE]` (or `[SKIPPED]` for
   A1.2 if applicable).
2. Verify all `[NOT DONE]` items in [`HUMAN_TASKS.md`](HUMAN_TASKS.md) are
   `[DONE]` per the human's reports.
3. Tell the human: "Plan Phase 01 closed. Recommended next:
   [`../02-LAUNCH-FULL-PHASE-1-RUN/`](../02-LAUNCH-FULL-PHASE-1-RUN/) (or
   [`../03-CPU-TO-GPU-OFFLOAD/`](../03-CPU-TO-GPU-OFFLOAD/) if A1.4 routed
   there)."

**How to verify:**

- All tasks in both TASKS.md and HUMAN_TASKS.md for this folder are `[DONE]`
  or explicitly `[SKIPPED]`/`[CANCELLED]` with a reason.
- The handoff message has been delivered.
