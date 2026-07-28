# Rebuild Something-Something V2 from the official distribution

This guide recreates the project's SSv2 raw bytes, official train/validation views, labels map, and
deterministic tiny subset after the old volume is gone.

It does **not** derive video bytes from labels or W&B. The official research-use distribution (or an
independent surviving copy) is mandatory. Keep it private and follow Qualcomm's current license.

## Expected end state

| Item | Exact expected value |
|---|---:|
| Raw VP9/WebM videos | 220,847 |
| Full train symlinks | 168,913 |
| Full validation symlinks | 24,777 |
| Tiny train symlinks | 4,002 |
| Tiny validation symlinks | 348 |
| Label classes | 174 |
| Native dataset FPS | 12 |

The 220,847 raw files include the unlabeled test split. This project builds training views only for
the official train and validation IDs. `data.py` does not consume the class label during training,
but the split records and generated `labels.json` are required to reproduce membership and the tiny
stratification.

## Stage 0 — choose the byte source

Use this priority:

1. A verified exact private backup of `/workspace/ssv2_raw`.
2. The two original official ZIPs and label package found on the Mac/external disk.
3. A fresh download from Qualcomm's official Something-Something V2 page.

If (1) exists, restore it with the backup/partial-recovery guides and begin at Stage 5. If neither
(2) nor (3) is available, SSv2 cannot be reconstructed. Do not substitute a third-party mirror while
claiming official old-volume identity.

## Stage 1 — download every official package on the Mac

On the Mac, create a dedicated folder:

```bash
export SSV2_DOWNLOAD_DIR="$HOME/Downloads/HJEPA_SSV2_OFFICIAL_2026_07"
mkdir -p "$SSV2_DOWNLOAD_DIR"
printf '%s\n' "$SSV2_DOWNLOAD_DIR"
```

In a browser:

1. Open [Qualcomm's official Something-Something V2 download page](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads).
2. Sign in with the user's Qualcomm ID and accept the current research-use agreement if prompted.
3. Download **Something-Something dataset download instructions**.
4. Download both links under **Video Files**: `Something-something_zip1` and
   `Something-something_zip2`.
5. Download **Something-Something download package labels**.
6. Save all four items into the exact directory printed above. If the browser always saves to
   `~/Downloads`, move them in Finder only after all downloads finish.

Qualcomm's current page says to download **all** files. Its official PDF describes a 19.4 GB TGZ
archive split into two ZIP-wrapped parts and gives the extraction order. Do not begin while a
`.download`, `.crdownload`, or `.part` file exists.

List the received files and identify their real types:

```bash
find "$SSV2_DOWNLOAD_DIR" -maxdepth 1 -type f -print -exec file {} \;
```

The expected video archive filenames from the official instructions are:

```text
20bn-something-something-v2-00.zip
20bn-something-something-v2-01.zip
```

The labels download is a separate ZIP and the instructions are a PDF. If the browser assigned a
different filename, use `file` and the download page—not guesswork—to identify it. Do not rename or
delete either large archive merely because its displayed link text was `zip1`/`zip2`.

Require exactly two official video ZIPs:

```bash
test "$(find "$SSV2_DOWNLOAD_DIR" -maxdepth 1 -type f -name '20bn-something-something-v2-??.zip' | wc -l | tr -d ' ')" = "2"
```

Create a local checksum manifest. These SHA-256 values are the user's download identities; do not
mislabel them as publisher-supplied checksums:

```bash
(cd "$SSV2_DOWNLOAD_DIR" && find . -maxdepth 1 -type f ! -name DOWNLOADS_SHA256.txt -exec shasum -a 256 {} \; > DOWNLOADS_SHA256.txt)
```

Verify the manifest immediately:

```bash
(cd "$SSV2_DOWNLOAD_DIR" && shasum -a 256 -c DOWNLOADS_SHA256.txt)
```

## Stage 2 — upload the packages directly to the new volume

If following the ground-up master, the `newrunpod:` in-memory rclone remote and
`$NEW_RUNPOD_VOLUME_ID` already exist. Confirm them:

```bash
rclone listremotes
printf '%s\n' "$NEW_RUNPOD_VOLUME_ID"
```

Upload all packages to a staging directory without launching a Pod:

```bash
rclone copy "$SSV2_DOWNLOAD_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/ssv2_raw/downloads" --size-only --transfers 4 --checkers 8 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --stats 30s -P
```

Compare the Mac and new volume byte streams:

```bash
export SSV2_UPLOAD_CHECK_REPORT="${SSV2_DOWNLOAD_DIR}-check-new-runpod-downloads.txt"
rclone check "$SSV2_DOWNLOAD_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/ssv2_raw/downloads" --download --checkers 4 --combined "$SSV2_UPLOAD_CHECK_REPORT" -P
```

Require exit zero and only `=` lines. The report is outside the compared source directory so it
cannot create a false missing-file result. Copy the report to the volume's separate evidence prefix:

```bash
rclone copyto "$SSV2_UPLOAD_CHECK_REPORT" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-input/transfer-evidence/ssv2-download-check.txt"
```

Keep the Mac originals.

If a Pod is already running and direct S3 is unavailable, full SSH with a public IP supports SCP;
basic proxied RunPod SSH does not. Do not pay an idle Pod for hours if the S3 upload works.

## Stage 3 — locate and verify packages on the Pod

Run after the new volume is mounted at `/workspace` and the fresh Pod has `unzip` installed:

```bash
find /workspace/ssv2_raw/downloads -maxdepth 1 -type f -printf '%s\t%f\n' | sort
```

Verify the Mac checksum manifest. Its entries are relative to the download directory:

```bash
cd /workspace/ssv2_raw/downloads
sha256sum -c DOWNLOADS_SHA256.txt
```

`sha256sum` accepts the checksum-file format written by macOS `shasum`. Require every item to print
`OK`.

Validate both video ZIP archives without extracting:

```bash
unzip -t /workspace/ssv2_raw/downloads/20bn-something-something-v2-00.zip
unzip -t /workspace/ssv2_raw/downloads/20bn-something-something-v2-01.zip
```

Do not continue if either integrity test fails. Redownload only the bad archive, recompute the Mac
manifest, re-upload, and recheck.

## Stage 4 — unwrap and extract the official video archive

Create fresh staging/output directories:

```bash
mkdir -p /workspace/ssv2_raw/parts
mkdir -p /workspace/ssv2_raw/labels
```

Unzip the two wrappers without overwriting an existing part:

```bash
unzip -n /workspace/ssv2_raw/downloads/20bn-something-something-v2-00.zip -d /workspace/ssv2_raw/parts
unzip -n /workspace/ssv2_raw/downloads/20bn-something-something-v2-01.zip -d /workspace/ssv2_raw/parts
```

Require both concatenation parts and inspect their types/sizes:

```bash
test -s /workspace/ssv2_raw/parts/20bn-something-something-v2-00
test -s /workspace/ssv2_raw/parts/20bn-something-something-v2-01
file /workspace/ssv2_raw/parts/20bn-something-something-v2-00
file /workspace/ssv2_raw/parts/20bn-something-something-v2-01
du -h /workspace/ssv2_raw/parts/20bn-something-something-v2-0?
```

Concatenate in numeric order and stream-extract the TGZ, exactly following the publisher's format:

```bash
cat /workspace/ssv2_raw/parts/20bn-something-something-v2-00 /workspace/ssv2_raw/parts/20bn-something-something-v2-01 | tar -xzf - -C /workspace/ssv2_raw
```

Require the exact raw directory and file count:

```bash
test -d /workspace/ssv2_raw/20bn-something-something-v2
test "$(find /workspace/ssv2_raw/20bn-something-something-v2 -maxdepth 1 -type f -name '*.webm' | wc -l | tr -d ' ')" = "220847"
echo "SSv2 raw count: 220847 OK"
```

Record the extracted size and a sorted ID/size inventory before creating views:

```bash
du -sh /workspace/ssv2_raw/20bn-something-something-v2
find /workspace/ssv2_raw/20bn-something-something-v2 -maxdepth 1 -type f -name '*.webm' -printf '%f\t%s\n' | sort > /workspace/ssv2_raw/raw-id-size.tsv
```

Keep downloads and parts until the entire rebuilt volume and an independent backup pass validation.

## Stage 5 — extract and validate official labels/splits

Find the single label ZIP without dumping its contents:

```bash
find /workspace/ssv2_raw/downloads -maxdepth 1 -type f -iname '*label*.zip' -printf '%p\n'
```

Require one match and capture its path:

```bash
test "$(find /workspace/ssv2_raw/downloads -maxdepth 1 -type f -iname '*label*.zip' | wc -l | tr -d ' ')" = "1"
export SSV2_LABEL_ARCHIVE="$(find /workspace/ssv2_raw/downloads -maxdepth 1 -type f -iname '*label*.zip' -print -quit)"
test -s "$SSV2_LABEL_ARCHIVE"
```

Test and extract it:

```bash
unzip -t "$SSV2_LABEL_ARCHIVE"
unzip -n "$SSV2_LABEL_ARCHIVE" -d /workspace/ssv2_raw/labels
```

List the JSON filenames:

```bash
find /workspace/ssv2_raw/labels -type f -name '*.json' -printf '%p\n' | sort
```

Capture the current official train and validation files by their publisher names:

```bash
export SSV2_TRAIN_JSON="$(find /workspace/ssv2_raw/labels -type f -name 'something-something-v2-train.json' -print -quit)"
export SSV2_VALIDATION_JSON="$(find /workspace/ssv2_raw/labels -type f -name 'something-something-v2-validation.json' -print -quit)"
test -s "$SSV2_TRAIN_JSON"
test -s "$SSV2_VALIDATION_JSON"
```

Validate record shapes, counts, unique IDs, disjointness, and raw targets:

```bash
python3 - "$SSV2_TRAIN_JSON" "$SSV2_VALIDATION_JSON" <<'PY'
import json
import sys
from pathlib import Path

raw = Path('/workspace/ssv2_raw/20bn-something-something-v2')
splits = {}
for name, argument, expected in (
    ('train', sys.argv[1], 168_913),
    ('validation', sys.argv[2], 24_777),
):
    payload = json.loads(Path(argument).read_text())
    assert isinstance(payload, list), (name, type(payload).__name__)
    assert len(payload) == expected, (name, len(payload), expected)
    ids = []
    for index, record in enumerate(payload):
        assert isinstance(record, dict), (name, index)
        assert record.get('id') is not None, (name, index, 'missing id')
        assert record.get('template') is not None, (name, index, 'missing template')
        video_id = str(record['id'])
        assert (raw / f'{video_id}.webm').is_file(), (name, video_id, 'missing raw target')
        ids.append(video_id)
    assert len(set(ids)) == expected, (name, 'duplicate ids')
    splits[name] = set(ids)
    print(name, expected, 'records OK')
assert splits['train'].isdisjoint(splits['validation'])
print('train/validation disjoint and all raw targets present')
PY
```

## Stage 6 — build the exact full split symlink view

The recovery helper refuses missing targets, unsafe IDs, incorrect links, and regular-file
collisions. Run its dry-run from the reconstructed project tree:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 NETWORK_VOLUME_RECOVERY/scripts/rebuild_ssv2_from_splits.py --train-json "$SSV2_TRAIN_JSON" --validation-json "$SSV2_VALIDATION_JSON" --workspace /workspace --dry-run
```

Require 168,913 train records, 24,777 validation records, zero collisions, and no fatal error. Then
create the links and generated `labels.json`:

```bash
python3 NETWORK_VOLUME_RECOVERY/scripts/rebuild_ssv2_from_splits.py --train-json "$SSV2_TRAIN_JSON" --validation-json "$SSV2_VALIDATION_JSON" --workspace /workspace
```

Run strict check mode:

```bash
python3 NETWORK_VOLUME_RECOVERY/scripts/rebuild_ssv2_from_splits.py --train-json "$SSV2_TRAIN_JSON" --validation-json "$SSV2_VALIDATION_JSON" --workspace /workspace --check
```

Require exact link counts and labels-map count:

```bash
test "$(find /workspace/data/ssv2/train -maxdepth 1 -type l -name '*.webm' | wc -l | tr -d ' ')" = "168913"
test "$(find /workspace/data/ssv2/validation -maxdepth 1 -type l -name '*.webm' | wc -l | tr -d ' ')" = "24777"
python3 -c "import json; assert len(json.load(open('/workspace/data/ssv2/labels.json'))) == 193690; print('labels.json: 193690 OK')"
```

Prove that links are absolute, target the canonical raw root, and are not broken:

```bash
python3 - <<'PY'
import os
from pathlib import Path

root = Path('/workspace/data/ssv2')
expected = Path('/workspace/ssv2_raw/20bn-something-something-v2')
links = [*sorted((root / 'train').glob('*.webm')), *sorted((root / 'validation').glob('*.webm'))]
assert len(links) == 168_913 + 24_777
for path in links:
    assert path.is_symlink(), path
    text = os.readlink(path)
    assert text.startswith('/workspace/ssv2_raw/20bn-something-something-v2/'), (path, text)
    assert path.resolve().parent == expected, (path, path.resolve())
    assert path.resolve().is_file(), path
print(len(links), 'full SSv2 symlinks resolve to canonical raw root')
PY
```

## Stage 7 — build the deterministic tiny subset

Run the current repository helper with its exact project parameters:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 make_subset.py --data-root /workspace/data --train-per-class 23 --val-per-class 2 --seed 42
```

Require exact counts and a manifest:

```bash
test "$(find /workspace/data/ssv2_tiny/train -maxdepth 1 -type l -name '*.webm' | wc -l | tr -d ' ')" = "4002"
test "$(find /workspace/data/ssv2_tiny/validation -maxdepth 1 -type l -name '*.webm' | wc -l | tr -d ' ')" = "348"
test -s /workspace/data/ssv2_tiny/manifest.json
echo "SSv2 tiny counts and manifest OK"
```

Rerun the same command, then require unchanged counts. This proves idempotence for the selected
parameters:

```bash
python3 make_subset.py --data-root /workspace/data --train-per-class 23 --val-per-class 2 --seed 42
```

```bash
test "$(find /workspace/data/ssv2_tiny/train -maxdepth 1 -type l -name '*.webm' | wc -l | tr -d ' ')" = "4002"
test "$(find /workspace/data/ssv2_tiny/validation -maxdepth 1 -type l -name '*.webm' | wc -l | tr -d ' ')" = "348"
```

## Stage 8 — decode and dataloader validation

Decode deterministic samples from both full splits:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 - <<'PY'
from pathlib import Path
from decord import VideoReader, cpu

for split in ('train', 'validation'):
    paths = sorted((Path('/workspace/data/ssv2') / split).glob('*.webm'))
    assert paths
    for path in (paths[0], paths[len(paths) // 2], paths[-1]):
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        assert len(reader) > 0, path
        fps = float(reader.get_avg_fps())
        assert 10.0 <= fps <= 14.0, (path, fps)
        print(split, path.name, len(reader), f'{fps:.3f} fps')
print('SSv2 decode samples OK')
PY
```

Run the actual full and tiny loaders:

```bash
python3 - <<'PY'
import torch
from config import Config
from data import build_dataloader

for dataset in ('ssv2_tiny', 'ssv2'):
    cfg = Config()
    cfg.data.dataset = dataset
    context, target = next(iter(build_dataloader(cfg, 'train', batch_size=2)))
    assert tuple(context.shape) == (2, 8, 3, 256, 256), (dataset, context.shape)
    assert tuple(target.shape) == (2, 8, 3, 256, 256), (dataset, target.shape)
    assert torch.isfinite(context).all() and torch.isfinite(target).all()
    print(dataset, tuple(context.shape), float(context.min()), float(context.max()), 'OK')
PY
```

## Stage 9 — record new provenance and preserve the source packages

Create a dataset-build evidence directory:

```bash
export SSV2_BUILD_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "/workspace/preflight/ssv2-rebuild-${SSV2_BUILD_STAMP}"
cp /workspace/ssv2_raw/downloads/DOWNLOADS_SHA256.txt "/workspace/preflight/ssv2-rebuild-${SSV2_BUILD_STAMP}/"
cp /workspace/ssv2_raw/raw-id-size.tsv "/workspace/preflight/ssv2-rebuild-${SSV2_BUILD_STAMP}/"
cp /workspace/data/ssv2/labels.json "/workspace/preflight/ssv2-rebuild-${SSV2_BUILD_STAMP}/"
cp /workspace/data/ssv2_tiny/manifest.json "/workspace/preflight/ssv2-rebuild-${SSV2_BUILD_STAMP}/ssv2-tiny-manifest.json"
```

Hash the core records:

```bash
(cd "/workspace/preflight/ssv2-rebuild-${SSV2_BUILD_STAMP}" && sha256sum * > SHA256SUMS)
```

Do not delete the official ZIPs, concatenation parts, or label package yet. They are the cheapest
way to rebuild again and must be included in the new independent backup. Only consider removing
redundant parts after raw bytes, official packages, manifests, and the complete volume exist in at
least two verified independent locations.
