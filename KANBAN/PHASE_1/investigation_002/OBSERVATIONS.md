# Observations — Investigation 002 (dataloader throughput)

## Baseline (pre-fix)

200-step timed smoke on `ssv2_tiny`, default `log_every=50` (`efficient-aardvark-2`,
commit `4c1abb3` era — the `--log-every`/`--diag-every` CLI flag commit):

- **`real 5m31.226s` / 200 = 1.66 s/step** wall-clock on the A100 pod
- CPU saturated: `user 35m55s + sys 7m13s ≈ 43m` core-time over `5m31s` real ≈ **7.8×**
  core-time, i.e. all 8 dataloader workers pegged while the GPU waited on batches
- Root cause (found by re-reading `data.py`): `_read_video_decord` decoded **all**
  ~30–90 frames of each SSv2 clip, then sliced down to the ~16 actually needed
  (`t_ctx=8` × 2 clips) — ~80% of decode work thrown away. Compounded by
  `num_threads=1` (kept for VP9 `.webm` decode reliability).

Pod hardware (H1.3): 128 vCPU AMD EPYC 7742, threads/core=1, 2 TiB RAM — so CPU/RAM
are categorically not the ceiling; the bottleneck was **per-worker single-threaded
decode**, which is exactly what the refactor targets.

## Fix

`_read_video_decord` split into `_open_video_reader` (metadata only) +
`_decode_frames` (decode only the requested indices); `__getitem__` now decodes ~16
frames instead of ~30–90 (`commit 5e78caa`, "Decode only needed frames in dataloader").
Bit-identical frames to the old path. `num_threads=1` retained.

## Fix result

- **`charmed-haze-4`: `real 4m42.195s` / 200 = 1.41 s/step`** — a ~15% improvement,
  far short of the ~3× a naive 16/60 frame ratio would predict.
- Step-0 diagnostics bit-identical pre→post (`L_flow=2.8726…`, `L_var=0.4267…`,
  `grad_has_nan=0`) on the deterministic seed-42 diag batch → refactor confirmed
  behavior-preserving.
- The remaining cost is **decord seeking through VP9 keyframes**, not array
  allocation — a fundamentally harder, Plan Phase 03 (CPU→GPU decode) problem.

## Belief update / the decision actually made

Throughput was a **dataloader issue**, not a model issue. The KANBAN-01 decision tree
said `>1.0 s/step → detour through Plan Phase 03 (CPU→GPU offload)`, and 1.41 s/step
clears that bar — but the team **consciously overrode** the rule: "no point optimizing
throughput for a model whose acceptance gates haven't been verified." Phase 03 was
therefore **deferred, not cancelled**, to be revisited only after a clean Phase 1
acceptance lands. This is why investigation_003 / 005 ran on full SSv2 at ~1.4 s/step
rather than waiting on a GPU-decode rewrite.

## W&B smoke runs

| Run | Steps | log_every | Pre/post fix | Finding |
|---|---|---|---|---|
| `youthful-pond-1` (x4pwz33d) | 100 | 50 | pre | Pod + W&B + training loop OK (connectivity only) |
| `efficient-aardvark-2` (fz7ztfc8) | 200 | 50 | pre | **1.66 s/step**; CPU-bound; metrics baseline |
| `comfy-glade-3` (0mgmqxxi) | 200 | **1** | pre | dense per-step logging; diagnostics identical to aardvark |
| `charmed-haze-4` (gj8ypv0d) | 200 | 50 | **post** | **1.41 s/step**; metrics bit-identical → fix validated |

Source: chat lines ~482–2870 (`cursor_messages`); W&B configs/summaries verified via MCP.

## Conclusion

**CLOSED — solved by selective decode.** Pod-side timing confirmed a ~15% improvement
to 1.41 s/step. The aspirational ≤0.8 s/step KANBAN-01 target was **not** hit, but the
result was judged sufficient to proceed to full SSv2 training; deeper GPU-decode work
was deferred behind the acceptance gates.

## Cross-investigation

Enabled the expensive runs in [investigation_003](../investigation_003/) and
[investigation_005](../investigation_005/) on full SSv2.
