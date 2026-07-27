# Rebuild the project EGO4D corpus from licensed source

This guide creates a new EGO4D derivation after the old generated chunks and selection records are
gone. It follows the current repository's deterministic selector and chunker contract:

- official `video_540ss` canonical videos;
- about 210 selected source hours;
- source-UID-level 90/10 train/validation split;
- four hour-balanced download batches;
- non-overlapping four-second chunks;
- 12 FPS, H.264/libx264, CRF 27, `yuv420p`, no audio, shorter side 256;
- 48 frames per output chunk;
- tiny subset of 4,000 train and 350 validation symlinks, maximum 10 chunks/source video.

The output is scientifically usable under current code, but it is not called the recovered old byte
corpus. Exact H.264 bytes can vary with source version, FFmpeg build, and encoding environment even
when settings match.

The canonical project source is
[`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md). This recovery
chapter supplies the complete primary path; use the canonical guide's Stage 4 repair blocks if the
official CLI reports an unavailable UID.

## Stage 0 — understand license, time, disk, and cost

EGO4D is not anonymous-download data. Official July 2026 documentation says:

- execute the license agreement first;
- approval is typically about 48 hours;
- emailed AWS credentials expire after 14 days;
- credentials can be renewed;
- the CLI supports `--video_uid_file` and `video_540ss`;
- `video_540ss` is downscaled to 540 pixels on the short side;
- canonical videos are 30 FPS.

Project planning estimates:

| Work | Estimate, not SLA |
|---|---:|
| Full selected raw source across all batches | ~290 GB transferred |
| Raw present at one time | ~72 GB |
| Final generated chunks | ~50–90 GB |
| Four batch download+chunk cycle | ~2.5–6 hours |
| Setup/selection/validation | additional attended time |

The ground-up guide's 300 GB volume is designed for this sequential process. Never download all
four raw batches simultaneously. Keep at least 220 GB free before the first batch.

## Stage 1 — request or renew access in the browser

On the Mac:

1. Open [EGO4D Start Here](https://ego4d-data.org/docs/start-here/).
2. Follow its link to the license portal at `ego4d.dev`.
3. Sign in with the identity that will use the data.
4. Read and accept the current agreement as an individual unless an authorized institutional
   signatory is acting for the institution.
5. Submit the request.
6. Wait for the approval email.
7. Confirm the credentials are less than 14 days old before starting Stage 3.

Do not paste credentials into this repository, W&B, a KANBAN file, a RunPod environment variable
saved in a template, or `/workspace`. The temporary AWS credential file is placed in `/root/.aws` on
the Pod's container disk.

## Stage 2 — verify the Pod and persistent volume

SSH into the new Pod and run:

```bash
df -h /workspace
python3 --version
ffmpeg -version | head -1
```

Require:

- `/workspace` is the new 300 GB network volume;
- at least 220 GB is available before batch 1;
- Python is at least 3.10;
- FFmpeg exists.

Verify the current helper files compile:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 -m py_compile select_ego4d_uids.py chunk_ego4d.py make_ego4d_subset.py config.py data.py provenance.py
```

Enter or create a persistent tmux session:

```bash
tmux has-session -t ego4d_rebuild 2>/dev/null && tmux attach -t ego4d_rebuild || tmux new -s ego4d_rebuild
```

All long commands below run inside `ego4d_rebuild`. Detach with `Ctrl-B`, release, then `D`.

## Stage 3 — install the official CLI and configure time-limited credentials

Inside tmux:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 -m pip install --upgrade pip
python3 -m pip install ego4d
```

Verify the command and UID-file option:

```bash
command -v ego4d
ego4d --help | grep -E 'video_uid_file|datasets|output_directory'
```

Create the credentials file through hidden prompts. The secret is never a command argument or
terminal echo:

```bash
python3 - <<'PY'
from getpass import getpass
from pathlib import Path

access = input('EGO4D AWS access key ID: ').strip()
secret = getpass('EGO4D AWS secret access key: ').strip()
assert access and secret
root = Path.home() / '.aws'
root.mkdir(mode=0o700, parents=True, exist_ok=True)
path = root / 'credentials'
path.write_text(
    '[default]\n'
    f'aws_access_key_id = {access}\n'
    f'aws_secret_access_key = {secret}\n'
)
path.chmod(0o600)
print(path, 'written with mode 0600; secret not displayed')
PY
```

Verify only structure and permissions:

```bash
python3 - <<'PY'
from pathlib import Path
import stat

path = Path.home() / '.aws' / 'credentials'
text = path.read_text()
assert 'aws_access_key_id' in text and 'aws_secret_access_key' in text
assert stat.S_IMODE(path.stat().st_mode) == 0o600
print(path, 'structure and mode OK')
PY
```

## Stage 4 — download authoritative metadata and tier inventory

Create persistent directories:

```bash
mkdir -p /workspace/ego4d_raw
mkdir -p /workspace/ego4d_raw/manifests
mkdir -p /workspace/ego4d_raw/v2/video_540ss
mkdir -p /workspace/data/ego4d/train /workspace/data/ego4d/validation
```

Download official annotations plus top-level `ego4d.json`:

```bash
ego4d --output_directory /workspace/ego4d_raw --datasets annotations -y
```

The project's prior observed planning value was about 6 GB for this operation; the live CLI is the
authority. Require metadata:

```bash
test -s /workspace/ego4d_raw/ego4d.json
ls -lh /workspace/ego4d_raw/ego4d.json
```

Download the small authoritative v2.1 `video_540ss` UID manifest:

```bash
python3 - <<'PY'
from pathlib import Path
import boto3

out = Path('/workspace/ego4d_raw/video_540ss_manifest.csv')
boto3.session.Session(profile_name='default').client('s3').download_file(
    'ego4d-consortium-sharing',
    'public/v2_1/video_540ss/manifest.csv',
    str(out),
)
assert out.is_file() and out.stat().st_size > 0
print(out, out.stat().st_size, 'bytes OK')
PY
```

Inspect only its schema/count and hash both authoritative inputs:

```bash
python3 - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

metadata = Path('/workspace/ego4d_raw/ego4d.json')
tier = Path('/workspace/ego4d_raw/video_540ss_manifest.csv')
payload = json.loads(metadata.read_text())
assert isinstance(payload.get('videos'), list) and payload['videos']
with tier.open(newline='') as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames and 'video_uid' in reader.fieldnames, reader.fieldnames
    uids = {row['video_uid'].strip() for row in reader if row.get('video_uid', '').strip()}
assert uids
for path in (metadata, tier):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path, digest)
print('metadata videos:', len(payload['videos']))
print('video_540ss downloadable UIDs:', len(uids))
PY
```

## Stage 5 — generate a new deterministic selection and four batches

Because this is no-backup Case C, the old selection is lost. Generate a new one with current code:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 select_ego4d_uids.py \
  --metadata /workspace/ego4d_raw/ego4d.json \
  --download-manifest /workspace/ego4d_raw/video_540ss_manifest.csv \
  --target-hours 210 \
  --val-fraction 0.10 \
  --batches 4 \
  --seed 42 \
  --out-dir /workspace/ego4d_raw/manifests
```

Validate required files, uniqueness, partitioning, selection total, and both splits:

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path('/workspace/ego4d_raw/manifests')
required = [
    'train_uids.txt', 'val_uids.txt',
    'batch_1_uids.txt', 'batch_2_uids.txt',
    'batch_3_uids.txt', 'batch_4_uids.txt',
    'selection_manifest.json',
]
for name in required:
    path = root / name
    assert path.is_file() and path.stat().st_size > 0, path

def read(name):
    return [line.strip() for line in (root / name).read_text().splitlines() if line.strip()]

train = read('train_uids.txt')
validation = read('val_uids.txt')
batches = []
for index in range(1, 5):
    batch = read(f'batch_{index}_uids.txt')
    assert batch
    batches.extend(batch)
    print('batch', index, len(batch), 'source videos')
assert not set(train) & set(validation)
assert sorted(batches) == sorted(train + validation)
assert not [uid for uid, count in Counter(batches).items() if count != 1]
manifest = json.loads((root / 'selection_manifest.json').read_text())
assert manifest['batches'] == 4
assert manifest['totals']['hours'] >= 200
assert manifest['totals']['train_videos'] > 0
assert manifest['totals']['val_videos'] > 0
print('selection totals:', manifest['totals'])
print('selection and batch partition OK')
PY
```

Copy these small provenance files to an independent location as soon as practical. They define this
new derivation.

## Stage 6 — process batch 1

Set the current batch variables:

```bash
export BATCH=1
export UID_FILE="/workspace/ego4d_raw/manifests/batch_${BATCH}_uids.txt"
export RAW_DIR="/workspace/ego4d_raw/v2/video_540ss"
test -s "$UID_FILE" && echo "selected EGO4D batch ${BATCH}/4"
```

Download only those source videos:

```bash
ego4d --output_directory /workspace/ego4d_raw \
  --datasets video_540ss \
  --video_uid_file "$UID_FILE" \
  -y
```

Require the raw filenames to match the requested UID set exactly—not merely have the same count:

```bash
python3 - <<'PY'
import os
from pathlib import Path

uid_file = Path(os.environ['UID_FILE'])
raw_dir = Path(os.environ['RAW_DIR'])
requested = {line.strip() for line in uid_file.read_text().splitlines() if line.strip()}
found = {path.stem for path in raw_dir.glob('*.mp4') if path.is_file()}
missing = sorted(requested - found)
extra = sorted(found - requested)
assert requested, uid_file
assert not missing, ('missing raw UIDs', missing[:20], 'total', len(missing))
assert not extra, ('unexpected raw UIDs', extra[:20], 'total', len(extra))
print(len(found), 'raw files exactly match requested UID set')
PY
```

Decode-check a deterministic raw source:

```bash
python3 - <<'PY'
from pathlib import Path
from decord import VideoReader, cpu

paths = sorted(Path('/workspace/ego4d_raw/v2/video_540ss').glob('*.mp4'))
assert paths
path = paths[0]
reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
fps = float(reader.get_avg_fps())
assert len(reader) > 0
assert 29.0 <= fps <= 31.0, fps
print(path.name, len(reader), f'{fps:.3f} fps raw OK')
PY
```

Chunk the batch with the exact current contract:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 chunk_ego4d.py \
  --raw-dir /workspace/ego4d_raw/v2/video_540ss \
  --manifest /workspace/ego4d_raw/manifests/selection_manifest.json \
  --metadata /workspace/ego4d_raw/ego4d.json \
  --out-root /workspace/data/ego4d \
  --chunk-seconds 4 \
  --fps 12 \
  --shorter-side 256 \
  --crf 27 \
  --workers 2
```

Do not continue unless the final summary says `failed: 0`. If resource-related FFmpeg errors occur,
keep every raw file and rerun the idempotent command with `--workers 1`.

Decode-check three chunks per split:

```bash
python3 - <<'PY'
from pathlib import Path
from decord import VideoReader, cpu

for split in ('train', 'validation'):
    paths = sorted((Path('/workspace/data/ego4d') / split).glob('*.mp4'))
    assert paths, split
    probes = (paths[0], paths[len(paths) // 2], paths[-1])
    print(split, len(paths), 'cumulative chunks')
    for path in probes:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        height, width, _ = reader[0].shape
        fps = float(reader.get_avg_fps())
        assert len(reader) == 48, (path, len(reader))
        assert 11.5 <= fps <= 12.5, (path, fps)
        assert min(height, width) == 256, (path, width, height)
        print(path.name, len(reader), f'{fps:.3f}', f'{width}x{height}')
print('processed decode geometry OK')
PY
```

Inspect the cumulative manifest and require it is valid JSON:

```bash
test -s /workspace/data/ego4d/chunk_manifest.json
python3 -m json.tool /workspace/data/ego4d/chunk_manifest.json | sed -n '1,50p'
```

For batch 1 only, exercise the full data loader before deleting raw input:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 - <<'PY'
import torch
from config import Config
from data import build_dataloader

cfg = Config()
cfg.data.dataset = 'ego4d'
context, target = next(iter(build_dataloader(cfg, 'train', batch_size=2)))
assert tuple(context.shape) == (2, 8, 3, 256, 256), context.shape
assert tuple(target.shape) == (2, 8, 3, 256, 256), target.shape
assert torch.isfinite(context).all() and torch.isfinite(target).all()
print(tuple(context.shape), 'batch-1 EGO4D loader OK')
PY
```

Only after every batch-1 gate above succeeds, delete exactly batch 1's transient source videos:

```bash
python3 - <<'PY'
import os
from pathlib import Path

uid_file = Path(os.environ['UID_FILE'])
raw_dir = Path(os.environ['RAW_DIR'])
selected = [
    raw_dir / f'{uid}.mp4'
    for uid in (line.strip() for line in uid_file.read_text().splitlines())
    if uid
]
assert selected and all(path.is_file() for path in selected)
for path in selected:
    path.unlink()
leftovers = sorted(raw_dir.glob('*.mp4'))
assert not leftovers, ('unexpected raw files remain', [p.name for p in leftovers[:20]])
print('batch 1 raw inputs cleared after verification')
PY
```

The delete is intentional and space-bounded. If any verification failed, do not run it.

## Stage 7 — process batches 2, 3, and 4 sequentially

The block below is the complete gated sequence; there is no implied “repeat the earlier steps.” It
downloads, proves exact raw membership, decode-checks a source, chunks, decode-checks generated
outputs, validates the cumulative manifest, and only then deletes that batch's raw input. It uses
`set -euo pipefail`, so the loop stops at the first failed command and leaves that batch's raw files
in place.

Run the whole block inside `ego4d_rebuild`:

```bash
set -euo pipefail
export RAW_DIR=/workspace/ego4d_raw/v2/video_540ss

for BATCH in 2 3 4; do
    export BATCH
    export UID_FILE="/workspace/ego4d_raw/manifests/batch_${BATCH}_uids.txt"
    test -s "$UID_FILE"
    printf 'starting EGO4D batch %s/4\n' "$BATCH"

    ego4d --output_directory /workspace/ego4d_raw \
      --datasets video_540ss \
      --video_uid_file "$UID_FILE" \
      -y

    python3 - <<'PY'
import os
from pathlib import Path

uid_file = Path(os.environ['UID_FILE'])
raw_dir = Path(os.environ['RAW_DIR'])
requested = {line.strip() for line in uid_file.read_text().splitlines() if line.strip()}
found = {path.stem for path in raw_dir.glob('*.mp4') if path.is_file()}
missing = sorted(requested - found)
extra = sorted(found - requested)
assert requested, uid_file
assert not missing, ('missing raw UIDs', missing[:20], 'total', len(missing))
assert not extra, ('unexpected raw UIDs', extra[:20], 'total', len(extra))
print(len(found), 'raw files exactly match requested UID set')
PY

    python3 - <<'PY'
import os
from pathlib import Path
from decord import VideoReader, cpu

paths = sorted(Path(os.environ['RAW_DIR']).glob('*.mp4'))
assert paths
path = paths[0]
reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
fps = float(reader.get_avg_fps())
assert len(reader) > 0
assert 29.0 <= fps <= 31.0, fps
print(path.name, len(reader), f'{fps:.3f} fps raw OK')
PY

    cd /workspace/hierarchal-jepa-flow-world-model
    python3 chunk_ego4d.py \
      --raw-dir /workspace/ego4d_raw/v2/video_540ss \
      --manifest /workspace/ego4d_raw/manifests/selection_manifest.json \
      --metadata /workspace/ego4d_raw/ego4d.json \
      --out-root /workspace/data/ego4d \
      --chunk-seconds 4 \
      --fps 12 \
      --shorter-side 256 \
      --crf 27 \
      --workers 2

    python3 - <<'PY'
from pathlib import Path
from decord import VideoReader, cpu

for split in ('train', 'validation'):
    paths = sorted((Path('/workspace/data/ego4d') / split).glob('*.mp4'))
    assert paths, split
    probes = (paths[0], paths[len(paths) // 2], paths[-1])
    print(split, len(paths), 'cumulative chunks')
    for path in probes:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        height, width, _ = reader[0].shape
        fps = float(reader.get_avg_fps())
        assert len(reader) == 48, (path, len(reader))
        assert 11.5 <= fps <= 12.5, (path, fps)
        assert min(height, width) == 256, (path, width, height)
        print(path.name, len(reader), f'{fps:.3f}', f'{width}x{height}')
print('processed decode geometry OK')
PY

    test -s /workspace/data/ego4d/chunk_manifest.json
    python3 -m json.tool /workspace/data/ego4d/chunk_manifest.json >/dev/null

    python3 - <<'PY'
import os
from pathlib import Path

uid_file = Path(os.environ['UID_FILE'])
raw_dir = Path(os.environ['RAW_DIR'])
selected = [
    raw_dir / f'{uid}.mp4'
    for uid in (line.strip() for line in uid_file.read_text().splitlines())
    if uid
]
assert selected and all(path.is_file() for path in selected)
for path in selected:
    path.unlink()
leftovers = sorted(raw_dir.glob('*.mp4'))
assert not leftovers, ('unexpected raw files remain', [p.name for p in leftovers[:20]])
print(f"batch {os.environ['BATCH']} raw inputs cleared after verification")
PY

    printf 'completed EGO4D batch %s/4\n' "$BATCH"
done
```

If any command stops the loop, **do not** restart completed downloads. Read the last printed
`starting EGO4D batch N/4`, fix that batch, and rerun the same block with exactly this loop header:

| Failed/incomplete batch | Replacement header |
|---:|---|
| 2 | `for BATCH in 2 3 4; do` |
| 3 | `for BATCH in 3 4; do` |
| 4 | `for BATCH in 4; do` |

Existing generated chunks are idempotently skipped by `chunk_ego4d.py`. A failed batch's raw files
remain in place; rerunning the same batch lets the official CLI complete/verify its requested set,
after which the exact-membership gate catches any missing or foreign raw file. Never delete raw
files merely to make a gate pass.

If the CLI reports an unavailable `grp-*` or normal UUID, do not improvise. Execute the matching
**Repair Paste 2R** or **Repair Paste 2S** under Stage 4B of the
[canonical project guide](../../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md), require its success message,
then rerun the batch download. Current selector prefilters both conditions, so a failure suggests
source/version drift worth preserving in logs.

## Stage 8 — final full-corpus validation

After all four batches, verify expected selected source coverage, manifest agreement, count ranges,
and zero source-UID leakage:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 - <<'PY'
import json
from pathlib import Path

selection = json.loads(Path('/workspace/ego4d_raw/manifests/selection_manifest.json').read_text())
chunk_manifest = json.loads(Path('/workspace/data/ego4d/chunk_manifest.json').read_text())
root = Path('/workspace/data/ego4d')

expected = {'train': set(), 'validation': set()}
for item in selection['videos']:
    expected[item['split']].add(item['video_uid'])

actual = {}
counts = {}
for split in ('train', 'validation'):
    files = sorted((root / split).glob('*.mp4'))
    counts[split] = len(files)
    actual[split] = {path.name[:path.name.rfind('_')] for path in files}
    missing = expected[split] - actual[split]
    extra = actual[split] - expected[split]
    assert not missing, (split, 'missing', sorted(missing)[:10])
    assert not extra, (split, 'extra', sorted(extra)[:10])
    assert chunk_manifest['splits'][split]['chunks'] == counts[split]
    assert chunk_manifest['splits'][split]['source_uids'] == len(actual[split])
    print(split, counts[split], 'chunks from', len(actual[split]), 'source videos')

assert not actual['train'] & actual['validation']
assert 150_000 <= counts['train'] <= 190_000, counts['train']
assert 12_000 <= counts['validation'] <= 25_000, counts['validation']
print('EGO4D corpus, manifests, ranges, and split isolation OK')
PY
```

Require no transient raw MP4s and record final size:

```bash
test "$(find /workspace/ego4d_raw/v2/video_540ss -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')" = "0"
du -sh /workspace/data/ego4d
df -h /workspace
```

## Stage 9 — build and verify `ego4d_tiny`

Create the deterministic tiny subset:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 make_ego4d_subset.py \
  --data-root /workspace/data \
  --train-chunks 4000 \
  --val-chunks 350 \
  --per-video-cap 10 \
  --seed 42
```

Validate counts, links, and manifest:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path('/workspace/data/ego4d_tiny')
train = sorted((root / 'train').glob('*.mp4'))
validation = sorted((root / 'validation').glob('*.mp4'))
assert len(train) == 4000, len(train)
assert len(validation) == 350, len(validation)
assert all(path.is_symlink() and path.resolve().is_file() for path in train + validation)
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['splits']['train']['count'] == 4000
assert manifest['splits']['validation']['count'] == 350
print('ego4d_tiny 4000/350 links and manifest OK')
PY
```

Rerun the builder with identical arguments and require identical counts to prove idempotence.

## Stage 10 — record provenance and remove only expired credentials

Create a dated evidence package:

```bash
export EGO4D_BUILD_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}"
cp /workspace/ego4d_raw/ego4d.json "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/"
cp /workspace/ego4d_raw/video_540ss_manifest.csv "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/"
cp -R /workspace/ego4d_raw/manifests "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/selection-manifests"
cp /workspace/data/ego4d/chunk_manifest.json "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/"
cp /workspace/data/ego4d_tiny/manifest.json "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/ego4d-tiny-manifest.json"
ffmpeg -version > "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/ffmpeg-version.txt" 2>&1
python3 -m pip show ego4d > "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}/ego4d-package.txt"
```

Hash every evidence file recursively without including the hash file itself:

```bash
(cd "/workspace/preflight/ego4d-rebuild-${EGO4D_BUILD_STAMP}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
```

After all downloads and verification complete, remove the time-limited EGO4D credential file from
the disposable Pod container:

```bash
rm -f ~/.aws/credentials
```

This does not delete data. The credentials expire after 14 days regardless. Never copy the file into
`/workspace` or the independent backup.

## Stage 11 — current-project loader smoke

Run both EGO4D variants through the actual current loader:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 - <<'PY'
import torch
from config import Config
from data import build_dataloader

for dataset in ('ego4d_tiny', 'ego4d'):
    cfg = Config()
    cfg.data.dataset = dataset
    context, target = next(iter(build_dataloader(cfg, 'train', batch_size=2)))
    assert tuple(context.shape) == (2, 8, 3, 256, 256), (dataset, context.shape)
    assert tuple(target.shape) == (2, 8, 3, 256, 256), (dataset, target.shape)
    assert torch.isfinite(context).all() and torch.isfinite(target).all()
    print(dataset, tuple(context.shape), float(context.min()), float(context.max()), 'OK')
PY
```

Include the new EGO4D metadata/manifests and generated chunks in an independent backup before any
real training run. The transient raw sources may be reacquired while licensed, but the new selection
and exact generated output are now unique scientific state.
