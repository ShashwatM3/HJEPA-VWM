# Emergency copy: old RunPod volume to a Mac/external disk

Use this route when volume `4hzrwzk8ja` still exists but AWS signup/payment/activation cannot be
completed before deletion. It needs no running RunPod Pod and no AWS account. It does require a
valid RunPod S3 key, an awake Mac, reliable internet, and enough private local storage.

This route preserves every byte exposed by RunPod's object API. It does not magically turn the S3
view into a POSIX clone: symlinks may arrive as regular objects, and arbitrary owners, modes,
hardlinks, xattrs, and original timestamps are not guaranteed. The source inventories saved here
are what later reconstruct known directory and symlink topology.

## Stage 0 — choose the disk without erasing anything

Do not repartition, erase, or format a disk merely because this guide names APFS. First connect a
disk that is already expendable/empty or has a dedicated empty volume with enough free space.

Preferred destination properties:

- directly attached SSD/HDD, not an unreliable network share;
- encrypted, because SSv2/EGO4D and possibly credentials are private;
- case-sensitive APFS, because RunPod/Linux paths are case-sensitive;
- more free bytes than the source's S3-visible logical bytes plus at least 5% filesystem headroom;
- no FAT32, whose 4 GiB per-file limit can break large archives/checkpoints.

Ordinary case-insensitive APFS or exFAT is an emergency best-effort destination. Two source keys that
differ only by case can collide there. The final path/size and full-byte checks must decide whether
the copy is complete; never assume the filesystem is safe because it mounted.

List only external physical disks:

```bash
diskutil list external physical
```

Open Finder and identify the intended mounted volume. Set its exact mount path—this is an example,
so replace it:

```bash
export HJEPA_RESCUE_MOUNT="/Volumes/REPLACE_WITH_EXACT_EXTERNAL_VOLUME_NAME"
```

Require that it is a mounted directory and inspect its identity/free space:

```bash
test -d "$HJEPA_RESCUE_MOUNT"
diskutil info "$HJEPA_RESCUE_MOUNT"
df -h "$HJEPA_RESCUE_MOUNT"
```

Stop if `diskutil info` identifies the Mac system/data volume, a read-only filesystem, an unexpected
device, or a disk containing data you are not authorized to use.

Test case sensitivity using only a newly created temporary directory:

```bash
export HJEPA_CASE_TEST="$(mktemp -d "$HJEPA_RESCUE_MOUNT/.hjepa-case-test.XXXXXX")"
printf A > "$HJEPA_CASE_TEST/CaseProbe"
printf b > "$HJEPA_CASE_TEST/caseprobe"
find "$HJEPA_CASE_TEST" -mindepth 1 -maxdepth 1 -type f -print
```

Exactly two filenames must print for a case-sensitive destination. Confirm the count:

```bash
test "$(find "$HJEPA_CASE_TEST" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')" = "2"
```

If this returns nonzero, the filesystem is case-insensitive. Remove only this exact test directory
after inspecting it:

```bash
rm -f "$HJEPA_CASE_TEST/CaseProbe" "$HJEPA_CASE_TEST/caseprobe"
rmdir "$HJEPA_CASE_TEST"
unset HJEPA_CASE_TEST
```

If only one filename printed, the destination is case-insensitive. Prefer a different existing
case-sensitive volume. If the deletion deadline makes that impossible, continue only as a
best-effort rescue and require the later inventories to expose any collision.

## Stage 1 — create isolated data and control roots

Create one new rescue root. `volume/` will mirror the RunPod bucket; `recovery-manifests/` remains
outside it so checks compare only source data:

```bash
export LOCAL_SNAPSHOT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export LOCAL_RESCUE_ROOT="$HJEPA_RESCUE_MOUNT/HJEPA_RUNPOD_4hzrwzk8ja_RESCUE"
export LOCAL_VOLUME_DIR="$LOCAL_RESCUE_ROOT/volume"
export LOCAL_MANIFEST_DIR="$LOCAL_RESCUE_ROOT/recovery-manifests/$LOCAL_SNAPSHOT_ID"
export LOCAL_CONTROL_DIR="$HOME/Desktop/HJEPA_LOCAL_RESCUE_$LOCAL_SNAPSHOT_ID"
```

Require that no earlier rescue root will be mixed into this one:

```bash
test ! -e "$LOCAL_RESCUE_ROOT"
test ! -L "$LOCAL_RESCUE_ROOT"
mkdir -p "$LOCAL_VOLUME_DIR" "$LOCAL_MANIFEST_DIR" "$LOCAL_CONTROL_DIR"
```

Record the initial free bytes before copying:

```bash
python3 - <<'PY' > "$LOCAL_CONTROL_DIR/initial-disk-space.json"
import json
import os
import shutil

mount = os.environ['HJEPA_RESCUE_MOUNT']
usage = shutil.disk_usage(mount)
print(json.dumps({'mount': mount, 'total': usage.total, 'used': usage.used, 'free': usage.free}, indent=2))
PY
```

Record the source identity without credentials:

```bash
printf '%s\n' \
  'old_volume_id=4hzrwzk8ja' \
  'old_datacenter=US-MO-1' \
  'old_signing_region=us-mo-1' \
  'old_endpoint=https://s3api-us-mo-1.runpod.io' \
  "snapshot_id=$LOCAL_SNAPSHOT_ID" \
  "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$LOCAL_CONTROL_DIR/source-identity.txt"
```

## Stage 2 — install tools and prevent sleep

Install rclone and tmux if absent:

```bash
command -v brew
command -v rclone || brew install rclone
command -v tmux || brew install tmux
rclone version
tmux -V
```

Open a second Terminal window and run this command; leave that window open until Stage 10:

```bash
caffeinate -dimsu
```

Back in the first Terminal, start a persistent shell:

```bash
tmux new -s hjepa-local-rescue
```

All remaining long commands run inside this tmux session. Detach with `Ctrl-B`, release, then `D`.
Reattach with `tmux attach -t hjepa-local-rescue`.

## Stage 3 — create and configure a temporary RunPod S3 key

In the Mac browser:

1. Sign in to [RunPod](https://console.runpod.io/).
2. Open **Settings → S3 API Keys**—not ordinary API keys.
3. Create `temporary-local-volume-rescue-2026-07-23`.
4. Store the displayed `user_...` access ID and one-time `rps_...` secret in a password manager.
5. Do not put either value in this directory, Git, chat, screenshots, or command arguments.

Inside tmux, disable persistent rclone configuration:

```bash
export RCLONE_CONFIG=/dev/null
```

Read the access ID and hidden secret:

```bash
read -r -p "RunPod S3 access key (user_...): " RUNPOD_S3_ACCESS_KEY
read -r -s -p "RunPod S3 secret (rps_...): " RUNPOD_S3_SECRET_KEY; printf '\n'
```

Configure the source entirely in environment variables:

```bash
export RCLONE_CONFIG_RUNPOD_TYPE=s3
export RCLONE_CONFIG_RUNPOD_PROVIDER=Other
export RCLONE_CONFIG_RUNPOD_ENV_AUTH=false
export RCLONE_CONFIG_RUNPOD_ACCESS_KEY_ID="$RUNPOD_S3_ACCESS_KEY"
export RCLONE_CONFIG_RUNPOD_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET_KEY"
export RCLONE_CONFIG_RUNPOD_REGION=us-mo-1
export RCLONE_CONFIG_RUNPOD_ENDPOINT=https://s3api-us-mo-1.runpod.io
export RCLONE_CONFIG_RUNPOD_FORCE_PATH_STYLE=true
export RCLONE_CONFIG_RUNPOD_NO_CHECK_BUCKET=true
```

Prove that rclone sees the in-memory remote:

```bash
rclone listremotes
```

Expected output includes `runpod:`. Test only the source top level:

```bash
rclone lsf runpod:4hzrwzk8ja --max-depth 1 --log-file "$LOCAL_CONTROL_DIR/source-top-level.log" --log-level INFO
```

An authentication error does not prove deletion. Fix it with
[`03_TROUBLESHOOTING_BEFORE_DELETION.md`](03_TROUBLESHOOTING_BEFORE_DELETION.md). `NoSuchBucket`
after a valid key/endpoint/ID check requires immediate RunPod support escalation.

## Stage 4 — copy irreplaceable state first

Define non-deleting copy helpers. The destination path is always inside this rescue's `volume/`:

```bash
local_rescue_prefix() { local prefix="$1"; local log_name="${prefix//\//_}"; rclone copy "runpod:4hzrwzk8ja/${prefix}" "$LOCAL_VOLUME_DIR/${prefix}" --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$LOCAL_CONTROL_DIR/copy-${log_name}.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

```bash
local_rescue_file() { local object="$1"; local log_name="${object//\//_}"; rclone copyto "runpod:4hzrwzk8ja/${object}" "$LOCAL_VOLUME_DIR/${object}" --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$LOCAL_CONTROL_DIR/copy-${log_name}.log" --log-level INFO -P; }
```

Run these prefixes in order. A missing historical prefix can fail without invalidating the others;
authentication/read/write errors must be fixed:

```bash
local_rescue_prefix ckpt
local_rescue_prefix checkpoints
local_rescue_prefix stats
local_rescue_prefix preflight
local_rescue_prefix archive
local_rescue_prefix hierarchal-jepa-flow-world-model
local_rescue_prefix logs
local_rescue_prefix ego4d_raw/manifests
```

Copy small identity manifests individually:

```bash
local_rescue_file ego4d_raw/ego4d.json
local_rescue_file ego4d_raw/video_540ss_manifest.csv
local_rescue_file ego4d_raw/manifests/selection_manifest.json
local_rescue_file data/ssv2/labels.json
local_rescue_file data/ssv2_tiny/manifest.json
local_rescue_file data/ego4d/chunk_manifest.json
local_rescue_file data/ego4d_tiny/manifest.json
```

Copy the two tiny views:

```bash
local_rescue_prefix data/ssv2_tiny
local_rescue_prefix data/ego4d_tiny
```

## Stage 5 — measure source size and enforce the capacity gate

Now generate the complete source aggregate. It can be slow, but high-value files are already being
preserved:

```bash
rclone size runpod:4hzrwzk8ja --json > "$LOCAL_CONTROL_DIR/source-size.json"
python3 -m json.tool "$LOCAL_CONTROL_DIR/source-size.json"
```

Compare source logical bytes with the initial disk free bytes and require 5% headroom:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

control = Path(os.environ['LOCAL_CONTROL_DIR'])
source = json.loads((control / 'source-size.json').read_text())
disk = json.loads((control / 'initial-disk-space.json').read_text())
required = int(source['bytes'] * 1.05)
print('source objects:', source['count'])
print('source logical bytes:', source['bytes'])
print('initial destination free bytes:', disk['free'])
print('required with 5% headroom:', required)
assert disk['free'] >= required, 'disk is too small for a complete object-view rescue'
print('capacity gate PASS')
PY
```

If this fails, keep every copied file. Continue prioritizing checkpoints, repository state,
manifests, unique SSv2 raw video, and generated EGO4D chunks, but do not call the result complete.
Attach another suitable disk or switch to an activated cloud destination.

## Stage 6 — copy unique datasets and then the entire bucket

Copy the unique large roots first:

```bash
local_rescue_prefix ssv2_raw
local_rescue_prefix data/ego4d
local_rescue_prefix ego4d_raw
```

Define the whole-volume non-deleting pass:

```bash
local_rescue_all() { rclone copy runpod:4hzrwzk8ja "$LOCAL_VOLUME_DIR" --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$LOCAL_CONTROL_DIR/copy-all.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Run it and immediately print the status:

```bash
local_rescue_all
printf 'last rclone exit status: %s\n' "$?"
```

Inspect the last errors:

```bash
tail -n 100 "$LOCAL_CONTROL_DIR/copy-all.log"
```

Rerun the identical function until a complete pass exits zero, then run one additional complete
pass and require zero. Never change `copy` to `sync` and never add a delete flag.

## Stage 7 — inventory source and destination

Record destination aggregate size/count:

```bash
rclone size "$LOCAL_VOLUME_DIR" --json > "$LOCAL_CONTROL_DIR/destination-size.json"
```

Record path/size inventories:

```bash
rclone lsf runpod:4hzrwzk8ja -R --files-only --format sp --csv > "$LOCAL_CONTROL_DIR/source-size-path.csv"
rclone lsf "$LOCAL_VOLUME_DIR" -R --files-only --format sp --csv > "$LOCAL_CONTROL_DIR/destination-size-path.csv"
```

Record backend-visible time/size/path:

```bash
rclone lsf runpod:4hzrwzk8ja -R --files-only --format tsp --csv > "$LOCAL_CONTROL_DIR/source-time-size-path.csv"
rclone lsf "$LOCAL_VOLUME_DIR" -R --files-only --format tsp --csv > "$LOCAL_CONTROL_DIR/destination-time-size-path.csv"
```

Record directory paths separately, including source empty directories exposed by RunPod:

```bash
rclone lsf runpod:4hzrwzk8ja -R --dirs-only --format p --csv > "$LOCAL_CONTROL_DIR/source-directory-paths.csv"
rclone lsf "$LOCAL_VOLUME_DIR" -R --dirs-only --format p --csv > "$LOCAL_CONTROL_DIR/destination-directory-paths.csv"
```

Attempt MD5/size/path inventories. Empty/unsupported source hashes are metadata limitations, not
automatic corruption:

```bash
rclone lsf runpod:4hzrwzk8ja -R --files-only --hash MD5 --format shp --csv > "$LOCAL_CONTROL_DIR/source-md5-size-path.csv"
rclone lsf "$LOCAL_VOLUME_DIR" -R --files-only --hash MD5 --format shp --csv > "$LOCAL_CONTROL_DIR/destination-md5-size-path.csv"
```

Record tool/time information:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCAL_CONTROL_DIR/completed-at-utc.txt"
rclone version > "$LOCAL_CONTROL_DIR/rclone-version.txt"
sw_vers > "$LOCAL_CONTROL_DIR/macos-version.txt"
diskutil info "$HJEPA_RESCUE_MOUNT" > "$LOCAL_CONTROL_DIR/destination-disk-info.txt"
```

## Stage 8 — prove equality in increasing strength

Print source and destination aggregates:

```bash
python3 -m json.tool "$LOCAL_CONTROL_DIR/source-size.json"
python3 -m json.tool "$LOCAL_CONTROL_DIR/destination-size.json"
```

Counts and bytes must match. Run a complete path/size check:

```bash
rclone check runpod:4hzrwzk8ja "$LOCAL_VOLUME_DIR" --size-only --checkers 8 --combined "$LOCAL_CONTROL_DIR/check-size-combined.txt" --log-file "$LOCAL_CONTROL_DIR/check-size.log" --log-level INFO
LOCAL_SIZE_CHECK_STATUS="$?"
printf '%s\n' "$LOCAL_SIZE_CHECK_STATUS" > "$LOCAL_CONTROL_DIR/check-size-exit-status.txt"
test "$LOCAL_SIZE_CHECK_STATUS" -eq 0
```

Require exit zero and only `=` lines. Run the strongest full-byte comparison if the source remains:

```bash
rclone check runpod:4hzrwzk8ja "$LOCAL_VOLUME_DIR" --download --checkers 4 --combined "$LOCAL_CONTROL_DIR/check-download-combined.txt" --log-file "$LOCAL_CONTROL_DIR/check-download.log" --log-level INFO --stats 30s --stats-one-line -P
LOCAL_DOWNLOAD_CHECK_STATUS="$?"
printf '%s\n' "$LOCAL_DOWNLOAD_CHECK_STATUS" > "$LOCAL_CONTROL_DIR/check-download-exit-status.txt"
test "$LOCAL_DOWNLOAD_CHECK_STATUS" -eq 0
```

Require exit zero. If deletion is imminent, preserve the complete copy and size check even if the
full read cannot finish; document that limitation instead of claiming full verification.

## Stage 9 — freeze control evidence onto the external disk

Hash the control files without hashing the hash list itself:

```bash
(cd "$LOCAL_CONTROL_DIR" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | xargs -0 shasum -a 256 > SHA256SUMS)
```

Copy the complete control directory into the already separate manifest root:

```bash
rsync -a "$LOCAL_CONTROL_DIR/" "$LOCAL_MANIFEST_DIR/"
```

Verify the copied control hashes from the external disk:

```bash
(cd "$LOCAL_MANIFEST_DIR" && shasum -a 256 -c SHA256SUMS)
```

The external root must now contain exactly the `volume/` data tree and
`recovery-manifests/<SNAPSHOT_ID>/` evidence tree. Do not move manifest files into `volume/`.

## Stage 10 — revoke, duplicate, and eject safely

Unset in-memory secrets:

```bash
unset RUNPOD_S3_ACCESS_KEY RUNPOD_S3_SECRET_KEY RCLONE_CONFIG_RUNPOD_ACCESS_KEY_ID RCLONE_CONFIG_RUNPOD_SECRET_ACCESS_KEY
```

In RunPod **Settings → S3 API Keys**, revoke the temporary local-rescue key. In the second Terminal,
stop `caffeinate` with `Ctrl-C` only after all commands and hashes finish.

Keep the drive connected long enough to copy the entire rescue root to a second independent private
disk/location. One external disk is a rescue, not a durable backup. Licensed data must remain private.

Before ejecting, leave the mounted filesystem and flush pending writes:

```bash
cd "$HOME"
sync
diskutil eject "$HJEPA_RESCUE_MOUNT"
```

Wait for Finder/macOS to report successful ejection before unplugging. Label the physical disk with
volume ID `4hzrwzk8ja`, snapshot ID, UTC date, and whether the full-byte check passed—never with a
secret. Later restore it using
[`../PERSPECTIVE_2_AFTER_DELETION/02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md`](../PERSPECTIVE_2_AFTER_DELETION/02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md).
