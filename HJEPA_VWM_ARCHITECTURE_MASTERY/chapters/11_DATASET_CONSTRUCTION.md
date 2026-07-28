# 11 — Dataset construction

The runtime loader expects a common directory shape, but SSv2 and EGO arrive through fundamentally
different manufacturing pipelines.

## Common runtime shape

```text
<dataset_root>/
├── train/
└── validation/
```

Items may be `.webm` or `.mp4`. SSv2 directories contain symlinks to already-short raw videos.
Full EGO directories contain newly encoded four-second files; EGO tiny contains symlinks to those
chunks.

## Volume layout

```text
/workspace/
├── hierarchal-jepa-flow-world-model/   code only
├── data/
│   ├── ssv2/
│   ├── ssv2_tiny/
│   ├── ego4d/
│   └── ego4d_tiny/
├── ssv2_raw/
├── ego4d_raw/
├── hf_cache/
├── ckpt/<run_tag>/                     active per-run checkpoints
├── checkpoints/                        legacy default location
└── archive/
```

Code, data, caches, and run outputs are siblings. Real runs override the legacy config checkpoint
default with `/workspace/ckpt/<unique_tag>`.

## Full SSv2

The prepared full view contains:

| Split | Symlinks |
|---|---:|
| train | 168,913 |
| validation | 24,777 |

Symlinks resolve into `/workspace/ssv2_raw/20bn-something-something-v2/`. `labels.json` maps video
identity to class label. The pipeline does not re-encode these source clips.

## SSv2 tiny

`make_subset.py`:

1. loads full labels;
2. groups readable split symlinks by class;
3. sorts and deterministically shuffles with seed 42;
4. selects up to 23 training and two validation items per class;
5. resolves original links and creates idempotent tiny links directly to raw files;
6. writes `manifest.json` with selections, seed, counts, and undersized classes;
7. asserts every created link resolves.

With all 174 classes sufficiently populated:

```text
train = 174*23 = 4002
validation = 174*2 = 348
```

The common “~4,000/~350” description is approximate; 4,002/348 are the exact expected counts.

## EGO source selection

`select_ego4d_uids.py` reads official `ego4d.json` plus the authoritative `video_540ss` download-tier
CSV. It requires `video_uid` and `duration_sec`, then filters:

- v2.1 `grp-*` grouped aggregates;
- UIDs absent from `video_540ss`;
- stereo captures;
- videos with recorded FPS farther than 0.1 from 30;
- duration below 60 seconds.

Missing `is_stereo` is recorded and assumed mono. Missing FPS is reported/retained rather than
silently fabricated.

The committed corpus target is:

```text
target source duration = 210 hours
validation fraction    = 0.10 by source hours
download batches       = 4
seed                   = 42
primary-scenario cap   = 10% of target hours
```

Selection begins with a seeded sorted order and caps each primary scenario. If the cap prevents
reaching the requested duration, a warned relaxed pass tops up. Therefore diversity is a preference
with an explicit fallback, not a hard guarantee.

## Source-level split

Whole source videos accumulate into validation until approximately 10% of selected hours is reached;
all remaining sources go to training. The split shuffle uses `seed+1`.

Chunks from one source can never cross splits. A chunk-level random split would leak near-duplicate
temporal content and is prohibited.

## Download batches

Selected sources are assigned to four hour-balanced batches using longest-processing-time greedy
assignment, separately iterating train and validation members. Each whole source belongs to one
batch. This supports:

```text
download batch → chunk/verify → delete raw batch → next batch
```

It limits transient raw-volume use without changing final split identity.

Selection outputs:

```text
train_uids.txt
val_uids.txt
batch_1_uids.txt ... batch_4_uids.txt
selection_manifest.json
```

The manifest binds source metadata and download-tier CSV hashes plus per-UID duration, scenario,
split, and batch.

## EGO chunking

`chunk_ego4d.py` turns each selected long `video_540ss` source into non-overlapping:

```text
4.0-second, 12-fps, short-side-256 H.264 MP4 chunks
```

Partial tail windows are dropped. Any window intersecting a privacy redaction interval is skipped.
Output name:

```text
<video_uid>_<window_index:05d>.mp4
```

Encoding recipe:

| Setting | Value |
|---|---|
| video codec | `libx264` |
| preset | `veryfast` |
| CRF | 27 |
| pixel format | `yuv420p` |
| output FPS | 12 |
| short side | 256, other side even |
| GOP/keyframe interval | 12 frames = one second |
| audio | removed |
| container | fast-start MP4 |
| ffmpeg threads/worker | 1 |
| process workers | at most 4 |

30→12 FPS is a non-integer 2.5× resampling, alternating source-frame gaps of roughly two/three
frames. The model then samples stride two at 12 FPS, or one model frame every 1/6 second.

## Chunker durability

- Requires `ffmpeg` before worker creation.
- Only processes selected UIDs whose raw file is present; missing batch UIDs are pending.
- Existing non-empty outputs are skipped, making reruns idempotent.
- Writes `<stem>.part.mp4`; only a successful encode renames to final.
- A failed encode removes its partial and makes the batch command fail.
- `chunk_manifest.json` is rebuilt by rescanning the cumulative output tree after every invocation.

Raw source deletion is allowed only after chunking and verification succeed. The selection/download
manifests and `ego4d.json` are retained.

## Full EGO counts

The volume guide estimates roughly:

```text
train      165k–175k chunks
validation 15k–20k chunks
```

These are planning ranges, not architectural constants. The only exact count for a built corpus is
the verified manifest plus filesystem identity.

## EGO tiny

`make_ego4d_subset.py` creates:

```text
train      exactly 4,000 chunk symlinks
validation exactly 350 chunk symlinks
maximum    10 chunks per source UID
```

It writes a deterministic manifest. The per-source cap broadens coverage and limits adjacent chunks
from one long recording, but lexical diagnostic batching can still select same-source neighbors
unless explicitly diversified.

## Dataset identity versus builder manifest

A builder manifest says what was intended and produced. Runtime provenance independently scans what
actually exists:

- sorted paths;
- resolved sizes;
- Decord frame counts;
- aggregate counts/ranges;
- manifest hashes.

Both are needed. A stale manifest with missing files, or a silently changed file behind the same
symlink, changes the runtime fingerprint.

## Real-time interpretation

For 12 FPS EGO chunks with stride two:

```text
within-clip sample interval = 2/12 = 0.1667 s
eight sampled frames span   = 14/12 = 1.1667 s between first and last
default k=4 shift           = 4/12 = 0.3333 s
k=16 shift                  = 16/12 = 1.3333 s
```

For SSv2, real time depends on source FPS/container timing; `k` is fundamentally an index offset in
the decoded frame stream.

## Construction failure modes

- Downloading metadata UIDs not present in the selected tier.
- Chunk-level rather than source-level split.
- Deleting raw EGO before checking all worker exit states.
- Treating approximate full-corpus counts as identity.
- Moving data into the Git repository.
- Re-encoding SSv2 unnecessarily.
- Broken symlinks that still make directory counts look plausible.
- Ignoring privacy redaction intervals.
- Parallel ffmpeg oversubscription.
- Claiming dataset completeness without source-UID and manifest cross-checks.
