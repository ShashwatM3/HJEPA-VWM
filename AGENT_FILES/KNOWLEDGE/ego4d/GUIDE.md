# GUIDE — from today's SSv2-only volume to `--data ego4d` working end to end

> **What this is.** A sequential, follow-to-the-dot operator guide that takes the project from
> the SSv2-only starting point described in
> [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../../SETUPS/VOLUME_LAYOUT.md), through the Stage 2
> EGO4D code package, to a state where switching the CLI flag
> `--data ssv2 | ssv2_tiny | ego4d | ego4d_tiny` is the *only* thing that changes between an
> SSv2 experiment and an EGO4D experiment, and every experiment runs correctly on either.
>
> **How to use it.** Stages are strictly ordered; each ends with a **Verify** block — do not
> proceed past a failed verification. Stages marked `[DASHBOARD]` are browser clicks. Stages
> marked `[LOCAL]` are commands on your Mac/local checkout. Stages marked `[POD]` are commands
> inside SSH on the RunPod pod. Commands are copy-paste blocks. Replace only the explicit
> `PASTE_...` placeholders.
>
> **Facts baseline.** Every dataset statistic used here is from
> [`EGO4D_dataset_understanding.md`](EGO4D_dataset_understanding.md) (verified against
> ego4d-data.org, the CVPR paper, and the `facebookresearch/Ego4d` CLI README as of 2026-07):
> EGO4D canonical video is **30 FPS**, VP9, MP4 container; the `video_540ss` download tier is
> the full corpus downscaled to **540 px on the shorter side** (~5 TB for all 3,670 h ⇒
> **~1.4 GB per hour**); videos are long-form (average ~24 min, up to 7 h); access requires a
> signed license (~48 h approval, AWS credentials valid **14 days**). SSv2, per this repo's own
> pipeline, is 12 FPS, 240 px shorter side, VP9 `.webm`, 168,913 train / 24,777 validation
> clips of 2–6 s each.

---

> **Time budget at a glance.** Active human time end-to-end is roughly **1.5–2.5 hours**;
> machine time is dominated by the Stage 3→4 batched download-and-chunk loop (**~2.5–6 h
> unattended total** across the four sequential batches); the calendar is dominated by
> Stage 1's ~48 h license approval, which runs in parallel with Stage 2. Per-stage estimates
> sit under each stage heading.

## Stage 0 — Decisions locked in by this guide (read once, don't revisit mid-execution)

> **Time: ~10 min** (reading only — the decisions are already made).

These were open items in [`EGO4D_DATASET_CHAT_CONTEXT.md`](EGO4D_DATASET_CHAT_CONTEXT.md); this
guide commits to one answer for each so there is exactly one path to follow.

| Decision | Choice | Why |
|---|---|---|
| fps handling | **Option A: re-encode chunks to 12 FPS during chunking** | The training config stays byte-identical to SSv2 (`frame_stride=2`, `horizon_k` values, all SSv2-calibrated thresholds carry over unchanged); the cost is one-time preprocessing instead of a recurring decode tax and a config fork. Option B (native 30 FPS) is in Appendix B only for the record. |
| Integration strategy | **Sibling dataset** (per [`EGO4D usage options.md`](EGO4D%20usage%20options.md) Option 1) | SSv2 stays untouched and permanently comparable; first EGO4D run is a clean single-variable A/B against run 037. |
| Download tier | **`video_540ss`** | Officially supported, ~2.8× smaller than `full_scale`, still 30 FPS, and 540 px ≥ our 256 px training resolution with margin. |
| Chunk shape | **4-second, non-overlapping, 12 FPS, 256 px shorter side, H.264 `.mp4`, no audio** | 4 s × 12 FPS = 48 frames per chunk — comfortably above the loader's window span ((8−1)×2 + k = 14+k frames; k=12 ⇒ 26 ≤ 48), and matching SSv2's 2–6 s clip regime. Non-overlapping windows over ~190 train-hours yield ≈170k chunks, matching SSv2-full's 168,913. H.264 (not VP9) because it encodes ~10× faster and decodes faster in decord even at `num_threads=1`. |
| Corpus size | **~210 source hours** (≈190 h train / ≈20 h validation after the split) | 190 h × 3,600 s ÷ 4 s ≈ 171k train chunks ≈ SSv2-full scale; 20 h ⇒ ≈18k val chunks (SSv2 val is 24.8k — same order). |
| Train/val split | **By source video UID, ~90/10 by hours, before chunking** | Chunks of one long video are near-duplicates; splitting by chunk would leak train content into validation. |
| Download/chunk sequencing | **Batched: 4 batches of ~72 GB raw each, looped as download → chunk → verify → delete that batch's raw** | Raw 540ss files are a transient intermediate (read once by the chunker, then dead weight). Batching caps peak disk at ~225 GB instead of ~460 GB, so a 300 GB network volume suffices permanently — RunPod volumes can be grown but never shrunk, so the peak sets the bill forever. The final corpus is byte-identical to a one-shot download. |
| Volume layout | `/workspace/ego4d_raw/` for the CLI download; `/workspace/data/ego4d/{train,validation}/` holding **real chunk files** (not symlinks — chunks are new encodes, unlike SSv2 where raw files are already clip-sized); `/workspace/data/ego4d_tiny/` as symlinks into `data/ego4d` + `manifest.json` | Mirrors the `ssv2_raw` → `data/ssv2` → `data/ssv2_tiny` relationship as closely as the re-encode step allows. |

**Disk budget (do the arithmetic against your volume before Stage 3):**

- Raw `video_540ss` subset, total across all batches: ~1.4 GB/h × 210 h ≈ **~290 GB** (note
  the chat-context note's earlier 100–200 GB estimate was optimistic — 5 TB/3,670 h is the
  authoritative rate). Under the batched loop only **~72 GB** of it is on disk at once
  because this guide downloads only one batch at a time.
- Chunked output: 190k × ~250–450 KB (H.264 CRF 27, 256 px, high-motion egocentric) ≈
  **~50–90 GB**, permanent.
- Peak simultaneous usage: existing volume contents (~50–60 GB) + one raw batch + accumulated
  chunks ≈ **~225 GB** ⇒ **resize the network volume to 300 GB** before Stage 3 (do it
  early; volumes cannot be shrunk later, so do not over-provision "to be safe").
- Each batch's raw files are deleted at the end of its loop iteration (Stage 4); keep
  `ego4d.json` and the `manifests/` directory forever — they are tiny and are the provenance
  record.

---

## Stage 1 — `[DASHBOARD]` Sign the EGO4D license

> **Time: ~5–10 min active; ~48 h calendar wait** (passive — Stage 2 runs entirely inside this
> window, so the wait costs zero project time if you start it first).

Do this in the browser:

1. Open `https://ego4d.dev/request/ego4d`.
2. If it redirects, stay on the redirected official EGO4D license page.
3. Choose the EGO4D license/request option, not Ego-Exo4D.
4. Fill in your name, email, institution, and intended research use.
5. Sign as an individual unless you know UC Davis wants an institutional signatory.
6. Submit the form.
7. Wait for the credentials email. The email contains an AWS access key ID and an AWS secret
   access key. The credentials expire 14 days after issue.

**Verify before Stage 3:**

1. You have the credentials email open.
2. It contains an `aws_access_key_id`.
3. It contains an `aws_secret_access_key`.
4. The email arrived less than 14 days ago.

---

## Stage 2 — `[LOCAL + CODE]` All code changes, then push them to the pod

> **Time: ~2–4 h total** — roughly 1–3 h of coding-agent runtime for the eight deltas (the two
> new scripts in Deltas 4–5 are the bulk of it) plus ~30–60 min of your review. Fits
> comfortably inside Stage 1's approval window.

If the code in this branch already contains `select_ego4d_uids.py`, `chunk_ego4d.py`, and
`make_ego4d_subset.py`, skip straight to **Stage 2B** below. If those files do not exist, hand
the Delta 1-8 spec below to a coding agent as one work order.

### Stage 2A — implementation work order for a coding agent

Everything is **additive**: no SSv2 code path, default, or test changes behavior. The agent
must follow `AGENT_FILES/AGENTS.md` (§2.2 doc-sync rules, §16 verification) and run
`pytest -q` plus `python -m py_compile` on every touched file before finishing.

### Delta 1 — `data.py`: generalize the video glob

`SSV2Dataset.__init__` currently indexes `sorted((self.root / split).glob("*.webm"))`
(`data.py:174`). Change it to index the sorted union of `*.webm` and `*.mp4` in that
directory (one combined `sorted(...)` over both globs, so ordering stays deterministic), and
update the `FileNotFoundError` message and the class/docstring wording ("SSv2 .webm
symlinks") to say video files generally. Nothing else in the file changes: EGO4D chunks are
plain videos of ≥ 48 frames, so `_window_indices`, `_decode_frames`, the resize/crop/jitter/
normalize chain, and `num_threads=1` (harmless for H.264, still required for SSv2's VP9) all
apply unchanged.

### Delta 2 — `config.py`: EGO4D dataset roots

In `DataConfig`, mirroring the existing `full_root`/`tiny_root` properties exactly: add
`ego4d_root` returning `<data_root>/ego4d` and `ego4d_tiny_root` returning
`<data_root>/ego4d_tiny`; add `"ego4d"` and `"ego4d_tiny"` branches to `dataset_root()`;
update the `dataset` field's comment (`ssv2 | ssv2_tiny | ego4d | ego4d_tiny`).

### Delta 3 — extend every `--data` CLI to the four choices

Four files carry `choices=["ssv2", "ssv2_tiny"]` and all four must gain the two new choices,
keeping `ssv2_tiny` as the default everywhere: `train.py:1091`, `whiten_stats.py:149`,
`rank_probe.py:242`, `drift_probe.py:778`. Additionally `drift_probe.py:361` has its **own**
`.webm`-only glob (independent of `data.py`) — apply the same webm+mp4 union there and fix its
error message.

### Delta 4 — new script `select_ego4d_uids.py`

Purpose: read the downloaded `ego4d.json` metadata (it will live at
`/workspace/ego4d_raw/v2/ego4d.json`, see Stage 3) and emit the UID files that drive both the
download and the split. Spec:

- CLI: `--metadata <path to ego4d.json>`, `--target-hours` (default 210),
  `--val-fraction` (default 0.10), `--batches` (default 4), `--seed` (default 42),
  `--out-dir` (default `ego4d_manifests/`).
- Read the metadata's video list. **First inspect the actual downloaded file to confirm field
  names before coding against them** — the documented fields of interest are `video_uid`,
  `duration_sec`, `scenarios` (list of activity labels), `is_stereo`, and per-video stream
  facts (fps etc.) which may sit in a nested video-metadata object. The script must fail
  loudly if an expected field is missing rather than guessing.
- Filter: drop `is_stereo == true` videos (side-by-side stereo frames would corrupt training);
  drop videos whose canonical fps is not 30 (tolerance ±0.1) if the field is present; drop
  videos shorter than 60 s.
- Select for scenario diversity with a deterministic greedy pass: shuffle the filtered videos
  with the seed, iterate, and accept a video unless its primary scenario already holds more
  than ~10% of the accumulated hours; stop when `--target-hours` is reached.
- Split the **accepted videos** (not chunks) into train/val at `--val-fraction` of the
  accumulated hours, seeded.
- Write to `--out-dir`: `train_uids.txt` and `val_uids.txt` (one UID per line —
  whitespace-delimited is what the Ego4D CLI's `--video_uid_file` expects);
  `batch_1_uids.txt` … `batch_N_uids.txt`, a partition of the full selection into
  `--batches` groups balanced by total hours (assign whole source videos, never split one
  across batches; each batch should contain both train- and val-split videos so the split
  ratio is roughly preserved per batch); and a `selection_manifest.json` recording seed,
  filters applied, per-scenario hour totals, per-UID duration/split/batch — the provenance
  record. The batch files drive the Stage 3→4 download-and-chunk loop.

### Delta 5 — new script `chunk_ego4d.py`

Purpose: turn the downloaded long-form 540ss videos into the SSv2-shaped chunk corpus. Spec:

- CLI: `--raw-dir` (default `/workspace/ego4d_raw/v2/video_540ss`), `--manifest`
  (the `selection_manifest.json` from Delta 4 — provides each UID's split), `--out-root`
  (default `/workspace/data/ego4d`), `--chunk-seconds` (default 4), `--fps` (default 12),
  `--shorter-side` (default 256), `--crf` (default 27), `--workers` (default: CPU count),
  `--metadata` (path to `ego4d.json`, for redacted-interval lookup).
- For each source video (parallelized with a process pool over videos, not over chunks):
  compute non-overlapping `[t, t+4s)` windows over the video duration, discard the final
  partial window, discard any window overlapping one of that video's `redacted_intervals`
  (privacy-redacted ranges; treat a missing/empty field as no redactions), and encode each
  surviving window with ffmpeg:
  - seek/trim: fast-seek to the window start, duration 4 s;
  - filters: constant-frame-rate resample to 12 FPS, then scale so the **shorter** side is 256
    with the other side preserved to an even pixel count (the scale expression must handle
    portrait sources — do not assume height is the shorter side);
  - encode: `libx264`, `preset veryfast`, the `--crf` value, `yuv420p` pixel format, keyframe
    interval 12 (one keyframe per second, for cheap random access in decord), **no audio**,
    `+faststart` so the container index is at the front (decord opens faster);
  - output name: `<video_uid>_<window_index:05d>.mp4` into
    `<out-root>/<train|validation>/` per the UID's split from the manifest.
- Idempotent: skip windows whose output file already exists and is non-empty, so the run can
  be resumed after interruption.
- Batch-friendly by construction: process exactly those manifest UIDs whose raw file is
  **present** in `--raw-dir`, and emit a warning (never an error) listing manifest UIDs with
  no raw file on disk — under the batched loop those are simply later batches. The final
  summary line must print how many manifest UIDs were processed vs. still pending, so each
  loop iteration confirms it chunked exactly its batch.
- After encoding, write `<out-root>/chunk_manifest.json` by **rescanning `--out-root`** (not
  by counting only this invocation's work), so the manifest stays cumulative and correct
  across the four batch invocations: per split, chunk counts, source-UID counts, total
  encoded seconds, and the exact ffmpeg parameter set used.
- Honesty note to encode in a comment: 30→12 FPS is a non-integer (2.5×) decimation, so the
  CFR resample selects source frames at alternating 2/3-frame steps (~83 ms jitter). This is
  accepted; it is far below the 167 ms sampling interval the model sees at stride 2.

### Delta 6 — new script `make_ego4d_subset.py` (the `ego4d_tiny` builder)

`make_subset.py` is SSv2-specific (it stratifies by `labels.json` classes, which EGO4D lacks),
so a sibling script: deterministically (seed 42) sample **source videos** from
`data/ego4d/train` and `data/ego4d/validation` and symlink a capped number of chunks per
source video (cap ~10) until reaching ~4,000 train and ~350 val chunk symlinks in
`<data_root>/ego4d_tiny/{train,validation}/` — matching `ssv2_tiny`'s scale so smoke-run
wall-clock expectations carry over. Idempotent reruns, `manifest.json` with seed and selected
chunk names, symlink-only (never copy), exactly like `make_subset.py`'s contract.

### Delta 7 — tests

Additive tests only: extend `tests/test_phase1_contract.py` with assertions that
`cfg.data.ego4d_root`/`ego4d_tiny_root` and the two new `dataset_root()` branches resolve
correctly under a `JEPA_DATA_ROOT` override (mirroring the existing ssv2 assertions at
`tests/test_phase1_contract.py:31-40`); add a small unit test that a dataset directory
containing a mix of dummy `.webm` and `.mp4` files is indexed completely and in deterministic
sorted order by the Delta-1 glob (constructing `SSV2Dataset` only indexes paths — no decode —
so empty placeholder files suffice).

### Delta 8 — doc sync (mandatory per `AGENTS.md` §2.2, same session as the code)

- `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`: add `ego4d_raw/`, `data/ego4d/`, `data/ego4d_tiny/`
  to the target tree, note the real-files-not-symlinks difference, and add verification
  commands (chunk counts per split).
- `GUIDES/CODEBASE_STRUCTURE.md`: add the three new scripts to the file map.
- `AGENT_FILES/AGENTS.md`: §3 repository map (new scripts), §5 data path (the dataset is now
  selectable among four roots; EGO4D chunks are 12 FPS H.264 mp4), §12 data defaults table
  (`dataset` choices).
- Do **not** touch human-owned docs (`README.md`, `latest_brief.md`, etc.).

### Stage 2B — local verification commands

Run this from your Mac/local checkout:

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
python -m py_compile \
  config.py data.py train.py whiten_stats.py rank_probe.py drift_probe.py \
  select_ego4d_uids.py chunk_ego4d.py make_ego4d_subset.py
pytest -q
python train.py --help | grep -F "ego4d"
python whiten_stats.py --help | grep -F "ego4d"
python rank_probe.py --help | grep -F "ego4d"
python drift_probe.py --help | grep -F "ego4d"
git status --short
```

Expected:

```text
pytest reports all tests passed.
Each grep prints a help line containing ego4d / ego4d_tiny.
git status shows the Stage 2 files and this guide edit.
```

### Stage 2C — commit and push the Stage 2 package

Run this from your Mac/local checkout after Stage 2B passes:

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
git add \
  config.py data.py train.py whiten_stats.py rank_probe.py drift_probe.py \
  select_ego4d_uids.py chunk_ego4d.py make_ego4d_subset.py \
  tests/test_phase1_contract.py \
  AGENT_FILES/AGENTS.md \
  AGENT_FILES/SETUPS/VOLUME_LAYOUT.md \
  GUIDES/CODEBASE_STRUCTURE.md \
  AGENT_FILES/KNOWLEDGE/ego4d/
git commit -m "Add EGO4D sibling dataset pipeline"
git push
```

**Verify before Stage 3:** `git push` completes successfully. The pod will pull this commit in
Stage 3.

---

## Stage 3 — `[DASHBOARD + POD]` One-time download setup

> **Time: ~25–35 min, mostly attended.** Space/credential setup ~15 min; metadata fetch
> ~5–10 min; UID selection &lt;5 min. The actual video downloads happen inside the Stage 4
> loop.

Prerequisites:

1. Stage 1 credentials email is less than 14 days old.
2. Stage 2 commit has been pushed.
3. Your RunPod network volume has been resized to 300 GB before the first download.

### Stage 3A — resize the RunPod network volume to 300 GB

Do this in the RunPod dashboard:

1. Open `https://www.runpod.io/console`.
2. Click **Storage** in the left sidebar.
3. Open **Network Volumes**.
4. Find the network volume attached to this project.
5. Click the volume row's **three-dot menu**.
6. Click **Edit** or **Resize**.
7. Set **Size** to `300` GB.
8. Click **Save** / **Update**.
9. Wait until the volume row shows `300 GB`.

### Stage 3B — SSH into the pod and pull Stage 2

Run this on your Mac:

```bash
ssh runpod-jepa
```

If `ssh runpod-jepa` is not configured, do this in the RunPod dashboard:

1. Click **Pods**.
2. Click the running pod for this project.
3. Click **Connect**.
4. Copy the exact SSH command RunPod shows.
5. Paste that command into your Mac terminal.

Then run this inside the pod SSH session:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git pull
python -m py_compile \
  config.py data.py train.py whiten_stats.py rank_probe.py drift_probe.py \
  select_ego4d_uids.py chunk_ego4d.py make_ego4d_subset.py
python train.py --help | grep -F "ego4d"
df -h /workspace
```

Expected from `df -h /workspace`: the mounted `/workspace` filesystem has at least `220G`
available before Stage 4.

### Stage 3C — enter tmux

Run this inside the pod SSH session:

```bash
tmux has-session -t ego4d 2>/dev/null && tmux attach -t ego4d || tmux new -s ego4d
```

All remaining Stage 3 and Stage 4 commands run inside that `ego4d` tmux session.

### Stage 3D — install the EGO4D CLI

Run this inside tmux:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python -m pip install --upgrade pip
python -m pip install ego4d
command -v ego4d
ego4d --help | head -40
```

### Stage 3E — write AWS credentials

Run this inside tmux, replacing the two `PASTE_...` placeholders with values from the EGO4D
email:

```bash
mkdir -p ~/.aws
cat > ~/.aws/credentials <<'EOF'
[default]
aws_access_key_id = PASTE_YOUR_ACCESS_KEY_ID_HERE
aws_secret_access_key = PASTE_YOUR_SECRET_ACCESS_KEY_HERE
EOF
chmod 600 ~/.aws/credentials
python - <<'EOF'
from pathlib import Path

path = Path.home() / ".aws" / "credentials"
text = path.read_text()
assert "PASTE_YOUR" not in text, "Replace the placeholder AWS keys before continuing."
assert "aws_access_key_id" in text
assert "aws_secret_access_key" in text
print(path, "OK")
EOF
```

### Stage 3F — download EGO4D metadata

Run this inside tmux:

```bash
mkdir -p /workspace/ego4d_raw
ego4d --output_directory /workspace/ego4d_raw --datasets annotations -y
test -s /workspace/ego4d_raw/v2/ego4d.json
ls -lh /workspace/ego4d_raw/v2/ego4d.json
```

### Stage 3G — select source UIDs and four batches

Run this inside tmux:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python select_ego4d_uids.py \
  --metadata /workspace/ego4d_raw/v2/ego4d.json \
  --target-hours 210 \
  --val-fraction 0.10 \
  --batches 4 \
  --seed 42 \
  --out-dir /workspace/ego4d_raw/manifests
```

**Verify Stage 3:**

Run this inside tmux:

```bash
python - <<'EOF'
import json
from collections import Counter
from pathlib import Path

root = Path("/workspace/ego4d_raw/manifests")
required = [
    "train_uids.txt",
    "val_uids.txt",
    "batch_1_uids.txt",
    "batch_2_uids.txt",
    "batch_3_uids.txt",
    "batch_4_uids.txt",
    "selection_manifest.json",
]
for name in required:
    path = root / name
    assert path.exists() and path.stat().st_size > 0, f"missing or empty: {path}"

def read_ids(name: str) -> list[str]:
    return [line.strip() for line in (root / name).read_text().splitlines() if line.strip()]

train = read_ids("train_uids.txt")
val = read_ids("val_uids.txt")
batches = []
for idx in range(1, 5):
    ids = read_ids(f"batch_{idx}_uids.txt")
    print(f"batch {idx}: {len(ids)} source videos")
    batches.extend(ids)

dupes = [uid for uid, count in Counter(batches).items() if count > 1]
assert not dupes, f"UID appears in more than one batch: {dupes[:10]}"
assert sorted(batches) == sorted(train + val), "batches do not exactly partition train+val"

manifest = json.loads((root / "selection_manifest.json").read_text())
print("selected videos:", manifest["totals"]["videos"])
print("selected hours:", manifest["totals"]["hours"])
print("train videos:", manifest["totals"]["train_videos"])
print("val videos:", manifest["totals"]["val_videos"])
print("top scenarios:")
for scenario, hours in sorted(
    manifest["scenario_hours"].items(), key=lambda item: item[1], reverse=True
)[:12]:
    print(f"  {hours:6.1f} h  {scenario}")
assert manifest["batches"] == 4
assert manifest["totals"]["hours"] >= 200
print("Stage 3 manifest checks OK")
EOF
```

---

## Stage 4 — `[POD]` The batch loop: download → chunk → verify → delete raw, ×4

> **Time: ~2.5–6 h unattended total + ~20 min of verification across the four iterations.**
> Per batch: download ~72 GB (~6 min at 200 MB/s, ~25 min at 50 MB/s) + chunking ~30–60 min at
> full worker parallelism. Chunking is the single most compute-intensive step of the whole
> guide (~190,000 ffmpeg encodes over ~210 h of source video across the batches). This guide
> uses the sequential batched path only: download one batch, chunk it, verify it, delete that
> batch's raw files, then move to the next batch. Do not pipeline the first run.

Run this whole block inside the `ego4d` tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p /workspace/ego4d_raw/v2/video_540ss
mkdir -p /workspace/data/ego4d/train /workspace/data/ego4d/validation

for BATCH in 1 2 3 4; do
  UID_FILE="/workspace/ego4d_raw/manifests/batch_${BATCH}_uids.txt"
  RAW_DIR="/workspace/ego4d_raw/v2/video_540ss"

  echo "============================================================"
  echo "EGO4D batch ${BATCH}/4: download"
  echo "============================================================"
  test -s "${UID_FILE}"
  ego4d --output_directory /workspace/ego4d_raw \
        --datasets video_540ss \
        --video_uid_file "${UID_FILE}" \
        -y

  RAW_COUNT="$(find "${RAW_DIR}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
  UID_COUNT="$(wc -l < "${UID_FILE}" | tr -d ' ')"
  echo "raw mp4 count: ${RAW_COUNT}"
  echo "batch UID count: ${UID_COUNT}"
  test "${RAW_COUNT}" -eq "${UID_COUNT}"

  python - "${BATCH}" <<'EOF'
import random
import sys
from pathlib import Path
from decord import VideoReader, cpu

batch = sys.argv[1]
paths = sorted(Path("/workspace/ego4d_raw/v2/video_540ss").glob("*.mp4"))
assert paths, f"batch {batch}: no raw mp4 files downloaded"
p = random.choice(paths)
r = VideoReader(str(p), ctx=cpu(0), num_threads=1)
fps = float(r.get_avg_fps())
print(f"batch {batch} raw spot check: {p.name} | {len(r)} frames | {fps:.2f} fps")
assert 29.0 <= fps <= 31.0, f"expected about 30 fps, got {fps}"
EOF

  echo "============================================================"
  echo "EGO4D batch ${BATCH}/4: chunk"
  echo "============================================================"
  python chunk_ego4d.py \
    --raw-dir /workspace/ego4d_raw/v2/video_540ss \
    --manifest /workspace/ego4d_raw/manifests/selection_manifest.json \
    --metadata /workspace/ego4d_raw/v2/ego4d.json \
    --out-root /workspace/data/ego4d

  echo "============================================================"
  echo "EGO4D batch ${BATCH}/4: verify chunks"
  echo "============================================================"
  python - "${BATCH}" <<'EOF'
import random
import sys
from pathlib import Path
from decord import VideoReader, cpu

batch = sys.argv[1]
root = Path("/workspace/data/ego4d")
for split in ("train", "validation"):
    files = sorted((root / split).glob("*.mp4"))
    assert files, f"batch {batch}: no chunks yet in {split}"
    print(f"batch {batch} cumulative {split}: {len(files)} chunks")
    for p in random.sample(files, min(3, len(files))):
        r = VideoReader(str(p), ctx=cpu(0), num_threads=1)
        h, w, _ = r[0].shape
        fps = float(r.get_avg_fps())
        print(f"  {p.name} | {len(r)} frames | {fps:.2f} fps | {w}x{h}")
        assert len(r) == 48, f"{p} has {len(r)} frames, expected 48"
        assert 11.5 <= fps <= 12.5, f"{p} fps {fps}, expected about 12"
        assert min(h, w) == 256, f"{p} shorter side is {min(h, w)}, expected 256"
EOF

  python -m json.tool /workspace/data/ego4d/chunk_manifest.json | head -40

  if [ "${BATCH}" = "1" ]; then
    echo "============================================================"
    echo "EGO4D batch 1 only: dataloader smoke before raw deletion"
    echo "============================================================"
    python - <<'EOF'
from config import Config
from data import build_dataloader

cfg = Config()
cfg.data.dataset = "ego4d"
loader = build_dataloader(cfg, "train", batch_size=2)
c, t = next(iter(loader))
print(c.shape, t.shape, float(c.min()), float(c.max()))
assert tuple(c.shape) == (2, 8, 3, 256, 256)
assert tuple(t.shape) == (2, 8, 3, 256, 256)
assert float(c.min()) < -1.0 or float(c.max()) > 1.0
EOF
  fi

  echo "============================================================"
  echo "EGO4D batch ${BATCH}/4: delete transient raw files"
  echo "============================================================"
  xargs -a "${UID_FILE}" -I{} rm -f "${RAW_DIR}/{}.mp4"
  LEFT="$(find "${RAW_DIR}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
  echo "raw mp4 files left after deleting batch ${BATCH}: ${LEFT}"
  test "${LEFT}" -eq 0
done
```

**Final verify after the loop:**

Run this inside the `ego4d` tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model

python - <<'EOF'
import json
from pathlib import Path

selection = json.loads(Path("/workspace/ego4d_raw/manifests/selection_manifest.json").read_text())
chunk_manifest = json.loads(Path("/workspace/data/ego4d/chunk_manifest.json").read_text())
root = Path("/workspace/data/ego4d")

expected = {"train": set(), "validation": set()}
for item in selection["videos"]:
    expected[item["split"]].add(item["video_uid"])

actual = {}
chunks = {}
for split in ("train", "validation"):
    files = sorted((root / split).glob("*.mp4"))
    chunks[split] = len(files)
    actual[split] = {p.name[: p.name.rfind("_")] for p in files}
    missing = sorted(expected[split] - actual[split])
    extra = sorted(actual[split] - expected[split])
    print(f"{split}: {chunks[split]} chunks from {len(actual[split])} source videos")
    assert not missing, f"{split} missing source UIDs: {missing[:10]}"
    assert not extra, f"{split} extra source UIDs: {extra[:10]}"
    assert chunk_manifest["splits"][split]["chunks"] == chunks[split]
    assert chunk_manifest["splits"][split]["source_uids"] == len(actual[split])

overlap = actual["train"] & actual["validation"]
assert not overlap, f"source UID leakage across train/validation: {sorted(overlap)[:10]}"
assert 150_000 <= chunks["train"] <= 190_000, chunks["train"]
assert 12_000 <= chunks["validation"] <= 25_000, chunks["validation"]
print("Stage 4 manifest/filesystem/leakage checks OK")
EOF

du -sh /workspace/data/ego4d
test "$(find /workspace/ego4d_raw/v2/video_540ss -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')" = "0"
echo "raw video_540ss mp4 files left: 0"
```

---

## Stage 5 — `[POD]` Build `ego4d_tiny`

> **Time: ~5–10 min** (symlink creation plus verification; no encoding, no copying).

Run this inside the `ego4d` tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model
python make_ego4d_subset.py \
  --data-root /workspace/data \
  --train-chunks 4000 \
  --val-chunks 350 \
  --per-video-cap 10 \
  --seed 42
```

**Verify Stage 5:**

Run this inside the `ego4d` tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model

python - <<'EOF'
import json
from pathlib import Path

root = Path("/workspace/data/ego4d_tiny")
train = sorted((root / "train").glob("*.mp4"))
val = sorted((root / "validation").glob("*.mp4"))
manifest_path = root / "manifest.json"
assert len(train) == 4000, len(train)
assert len(val) == 350, len(val)
assert manifest_path.exists() and manifest_path.stat().st_size > 0
broken = [p for p in train + val if not p.resolve().exists()]
assert not broken, broken[:10]
manifest = json.loads(manifest_path.read_text())
assert manifest["splits"]["train"]["count"] == 4000
assert manifest["splits"]["validation"]["count"] == 350
print("ego4d_tiny train symlinks:", len(train))
print("ego4d_tiny validation symlinks:", len(val))
print("ego4d_tiny manifest OK:", manifest_path)
EOF

python make_ego4d_subset.py \
  --data-root /workspace/data \
  --train-chunks 4000 \
  --val-chunks 350 \
  --per-video-cap 10 \
  --seed 42
```

---

## Stage 6 — `[POD]` Smoke-test the full switchability claim

> **Time: ~25–45 min.** Dataloader smoke ~2 min; Stage-0 sanity ~5–10 min (add ~10 min on a
> fresh pod for the encoder download); the 500-step `ego4d_tiny` run ~10–13 min at the
> project's historical ~1.4 s/step; the 100-step SSv2 regression smoke ~3 min.

This is the stage that proves the guide's goal: the flag is the only difference.

Run this inside the pod SSH session:

```bash
tmux has-session -t ego4d_smoke 2>/dev/null && tmux attach -t ego4d_smoke || tmux new -s ego4d_smoke
```

Then run this inside the `ego4d_smoke` tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model

python - <<'EOF'
from config import Config
from data import build_dataloader

cfg = Config()
cfg.data.dataset = "ego4d_tiny"
loader = build_dataloader(cfg, "train", batch_size=2)
c, t = next(iter(loader))
print(c.shape, t.shape, float(c.min()), float(c.max()))
assert tuple(c.shape) == (2, 8, 3, 256, 256)
assert tuple(t.shape) == (2, 8, 3, 256, 256)
assert float(c.min()) < -1.0 or float(c.max()) > 1.0
EOF

python train.py --stage0-only

python train.py --data ego4d_tiny --steps 500 \
  --checkpoint-dir /workspace/ckpt/ego4d_smoke

test -f /workspace/ckpt/ego4d_smoke/phase1_step500.pt

python train.py --data ssv2_tiny --steps 100 \
  --checkpoint-dir /workspace/ckpt/ssv2_regression_smoke

test -f /workspace/ckpt/ssv2_regression_smoke/phase1_step100.pt
echo "Stage 6 switchability smoke passed"
```

**Verify Stage 6:** the block prints `Stage 6 switchability smoke passed`. At this point the
goal state is reached: any experiment command in the repo runs on either dataset by changing
only `--data` and the per-run `--checkpoint-dir`.

---

## Stage 7 — Running real EGO4D experiments correctly (read before the first 15k-step run)

> **Time: ~15 min reading now; the first real 15k-step A/B run itself is ~6 h of GPU time**
> (the project's 15k-step runs have historically clocked ~1.37 s/step ≈ 5.7 h; H.264 chunks
> should decode no slower than SSv2's VP9).

### Stage 7A — create a KANBAN investigation stub

Run this on your Mac/local checkout before the first real run:

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
INV_DIR="$(python - <<'EOF'
from pathlib import Path

root = Path("KANBAN/PHASE_1")
nums = []
for path in root.glob("investigation_[0-9][0-9][0-9]*"):
    try:
        nums.append(int(path.name.split("_")[1]))
    except (IndexError, ValueError):
        pass
print(f"investigation_{max(nums) + 1:03d}")
EOF
)"
mkdir -p "KANBAN/PHASE_1/${INV_DIR}"
cat > "KANBAN/PHASE_1/${INV_DIR}/DESCRIPTION.md" <<'EOF'
# EGO4D sibling A/B

Purpose: first clean full-prediction A/B after adding EGO4D as a sibling dataset.

Control baseline: run 037 `soft-universe-37` on SSv2.

Single intended delta: `--data ego4d` instead of `--data ssv2`.

Launch command:

```bash
python train.py --data ego4d --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 --lambda-sigreg 5.0 --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 2000 --predict-residual \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/ego4d_run037_ab --log-every 50 --diag-every 500
```
EOF
cat > "KANBAN/PHASE_1/${INV_DIR}/OBSERVATIONS.md" <<'EOF'
# Observations

Pending run.
EOF
cat > "KANBAN/PHASE_1/${INV_DIR}/NEXT_STEPS.md" <<'EOF'
# Next Steps

Pending run.
EOF
echo "created KANBAN/PHASE_1/${INV_DIR}"
```

### Stage 7B — optional dataset probes before the first full run

Run this inside a pod tmux session if you want a quick frozen-encoder drift/rank read before
burning a 15k-step run:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model
python rank_probe.py \
  --data ego4d \
  --split validation \
  --probe-videos 64 \
  --seed 42 \
  --out-dir logs/drift_probe_ego4d \
  --plot on
python drift_probe.py \
  --data ego4d \
  --split validation \
  --probe-videos 64 \
  --seed 42 \
  --encoder-curve on \
  --latent-curve off \
  --out-dir logs/drift_probe_ego4d
```

### Stage 7C — launch the first full EGO4D A/B run

Run this inside the pod SSH session:

```bash
tmux has-session -t ego4d_real 2>/dev/null && tmux attach -t ego4d_real || tmux new -s ego4d_real
```

Then run this inside the `ego4d_real` tmux session:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --data ego4d --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 --lambda-sigreg 5.0 --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 2000 --predict-residual \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/ego4d_run037_ab --log-every 50 --diag-every 500
test -f /workspace/ckpt/ego4d_run037_ab/phase1_step15000.pt
```

Do this in the W&B dashboard after the run appears:

1. Open `https://wandb.ai`.
2. Open the `hjepa-vwm` project.
3. Click the newest run.
4. Open **Config**.
5. Confirm `data.dataset` is `ego4d`.
6. Confirm `train.horizon_k` is `12`.
7. Confirm `train.predict_residual` is `true`.
8. Open the metrics workspace/panels.
9. Watch `grad_norm`, `grad_skipped`, `agc_*`, `coarse_vs_copy_ratio`,
   `coarse_vs_batch_mean_ratio`, and `coarse_copy_loss`.

### Stage 7D — if a future EGO4D run uses whitening, generate EGO4D whitening stats first

Run this inside a pod tmux session before any `--whiten-features --data ego4d` training run:

```bash
set -euo pipefail
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p logs/whiten
python whiten_stats.py \
  --data ego4d \
  --split train \
  --max-batches 200 \
  --seed 42 \
  --out logs/whiten/whiten_stats_ego4d_train_seed42.pt \
  --device cuda
test -f logs/whiten/whiten_stats_ego4d_train_seed42.pt
```

Then pass that file to the future training command:

```bash
python train.py --data ego4d --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ego4d_train_seed42.pt \
  --checkpoint-dir /workspace/ckpt/PASTE_RUN_TAG
```

---

## Appendix A — Failure modes and what they mean

| Symptom | Likely cause / fix |
|---|---|
| CLI errors with credential/403 failures | AWS credentials expired (14-day window) — renew via the license FAQ, update `~/.aws/credentials`, rerun (resumable). |
| `No .webm or .mp4 files found` from a probe script | Delta 1/3's twin change in `drift_probe.py` was missed — its glob is independent of `data.py`. |
| Chunk decode returns ≠ 48 frames | ffmpeg trim landed on a stream edge; the chunker's idempotent rerun should re-encode flagged files; the loader's clamp makes stragglers non-fatal but they should be rare (< 0.1%). |
| Val metrics implausibly good on EGO4D | Check the Stage-4 split-disjointness gate first — source-video leakage is the classic cause. |
| First EGO4D run's `L_recon_*` incomparable to SSv2 runs | Expected — different substrate; and if whitening is on, also a different whitened space. Compare EGO4D runs to EGO4D runs. |

## Appendix B — Option B (native 30 FPS), recorded but not chosen

Stream-copy chunking (no re-encode; fast preprocessing, heavier every-epoch decode of 540 px
VP9/H.264 at 30 FPS) with config-side compensation: every frame-index quantity multiplies by
2.5 — `frame_stride` 2→5 (config edit; there is no CLI flag for stride) and `horizon_k`
scaled per experiment (the SSv2-empirical k=12 ⇒ k=30 at 30 FPS). Costs: a permanent config
fork between datasets, SSv2-calibrated thresholds no longer carry over, and 2.5× more frames
decoded per sample forever. That trade is why Option A is the committed path.
