# Restore a verified local/external-disk backup to a new RunPod volume

Use this after the old volume is deleted when the complete rescue produced by
[`../PERSPECTIVE_1_BEFORE_DELETION/04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md`](../PERSPECTIVE_1_BEFORE_DELETION/04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md)
survives. This path does not require AWS and therefore has no AWS storage, request, or egress bill.
It does require funds for the new RunPod volume and later validation Pod.

The local backup layout must be:

```text
HJEPA_RUNPOD_4hzrwzk8ja_RESCUE/
├── volume/                         old object-view data only
└── recovery-manifests/
    └── <SNAPSHOT_ID>/              inventories, checks, logs, tool/disk identity
```

If those roots are mixed together, separate nothing by guesswork. Preserve the disk and classify it
as a partial/noncanonical backup until its inventories reconcile.

## Stage 0 — mount read-write, identify, and verify the backup

Connect the rescue disk to the Mac. In Terminal, list external disks:

```bash
diskutil list external physical
```

Set the exact mounted path and rescue root, replacing only the volume name if needed:

```bash
export HJEPA_RESCUE_MOUNT="/Volumes/REPLACE_WITH_EXACT_EXTERNAL_VOLUME_NAME"
export LOCAL_RESCUE_ROOT="$HJEPA_RESCUE_MOUNT/HJEPA_RUNPOD_4hzrwzk8ja_RESCUE"
export LOCAL_VOLUME_DIR="$LOCAL_RESCUE_ROOT/volume"
```

Verify all three directories and inspect disk identity:

```bash
test -d "$HJEPA_RESCUE_MOUNT"
test -d "$LOCAL_RESCUE_ROOT"
test -d "$LOCAL_VOLUME_DIR"
test -d "$LOCAL_RESCUE_ROOT/recovery-manifests"
diskutil info "$HJEPA_RESCUE_MOUNT"
```

List available snapshot directories without modifying them:

```bash
find "$LOCAL_RESCUE_ROOT/recovery-manifests" -mindepth 1 -maxdepth 1 -type d -print | sort
```

Set the exact snapshot that belongs to the complete rescue:

```bash
read -r -p "Paste the full chosen snapshot-directory path: " LOCAL_SNAPSHOT_DIR
export LOCAL_SNAPSHOT_DIR
test -d "$LOCAL_SNAPSHOT_DIR"
```

Verify its control-file checksums:

```bash
(cd "$LOCAL_SNAPSHOT_DIR" && shasum -a 256 -c SHA256SUMS)
```

Read the original proof results:

```bash
python3 -m json.tool "$LOCAL_SNAPSHOT_DIR/source-size.json"
python3 -m json.tool "$LOCAL_SNAPSHOT_DIR/destination-size.json"
cat "$LOCAL_SNAPSHOT_DIR/check-size-exit-status.txt"
test ! -f "$LOCAL_SNAPSHOT_DIR/check-download-exit-status.txt" || cat "$LOCAL_SNAPSHOT_DIR/check-download-exit-status.txt"
tail -n 50 "$LOCAL_SNAPSHOT_DIR/check-size.log"
test ! -f "$LOCAL_SNAPSHOT_DIR/check-download.log" || tail -n 50 "$LOCAL_SNAPSHOT_DIR/check-download.log"
```

Classify it:

- **Verified complete:** source/destination count+bytes match, the size check exited zero, and the
  full-download check exited zero.
- **Complete-looking:** count+bytes and size check match, but full-download proof did not finish.
- **Partial:** any missing/different/error row, interrupted copy, or count/byte mismatch exists.

Only the first two follow this whole-backup guide. A partial disk uses
[`06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md).

## Stage 1 — remeasure the disk copy without changing it

Install rclone if this Mac no longer has the rescue-time installation:

```bash
command -v brew
command -v rclone || brew install rclone
rclone version
```

Create a restore-control directory on the Mac's internal disk:

```bash
export LOCAL_RESTORE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export LOCAL_RESTORE_CONTROL="$HOME/Desktop/HJEPA_LOCAL_RESTORE_$LOCAL_RESTORE_ID"
mkdir -p "$LOCAL_RESTORE_CONTROL"
```

Remeasure the current `volume/` tree:

```bash
rclone size "$LOCAL_VOLUME_DIR" --json > "$LOCAL_RESTORE_CONTROL/live-local-size.json"
python3 -m json.tool "$LOCAL_RESTORE_CONTROL/live-local-size.json"
```

Require it to match the stored destination count and bytes:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

old = json.loads((Path(os.environ['LOCAL_SNAPSHOT_DIR']) / 'destination-size.json').read_text())
live = json.loads((Path(os.environ['LOCAL_RESTORE_CONTROL']) / 'live-local-size.json').read_text())
for field in ('count', 'bytes'):
    assert live[field] == old[field], (field, old[field], live[field])
print('local backup object count and bytes still match rescue evidence')
PY
```

If this fails, stop. Do not “repair” the only disk in place.

## Stage 2 — choose restore mode and calculate capacity

Choose exactly one mode:

- **POSIX-aware (recommended):** omit the six materialized link-view trees during upload, then
  recreate their symlinks from the rescue inventory/manifests on the Pod.
- **Literal object:** upload every local object as a regular object. It is simpler but can duplicate
  hundreds of thousands of videos and consume much more space.

Measure the POSIX-aware byte/count requirement with the same six filters used during upload:

```bash
rclone size "$LOCAL_VOLUME_DIR" --json --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' > "$LOCAL_RESTORE_CONTROL/posix-aware-size.json"
python3 -m json.tool "$LOCAL_RESTORE_CONTROL/posix-aware-size.json"
```

For literal mode, use `live-local-size.json`. For either mode calculate 20% headroom:

```text
new volume GB = ceil(selected_bytes / 1,000,000,000 × 1.20)
```

Set the selected size file. For POSIX-aware mode:

```bash
export SELECTED_RESTORE_SIZE_JSON="$LOCAL_RESTORE_CONTROL/posix-aware-size.json"
```

For literal mode instead:

```bash
export SELECTED_RESTORE_SIZE_JSON="$LOCAL_RESTORE_CONTROL/live-local-size.json"
```

Run the exact calculator and record its result:

```bash
python3 - <<'PY' | tee "$LOCAL_RESTORE_CONTROL/calculated-capacity.txt"
import json
import math
import os
from pathlib import Path

payload = json.loads(Path(os.environ['SELECTED_RESTORE_SIZE_JSON']).read_text())
capacity = math.ceil(payload['bytes'] / 1_000_000_000 * 1.20)
print('selected objects:', payload['count'])
print('selected bytes:', payload['bytes'])
print('minimum new volume capacity with 20% headroom:', capacity, 'GB')
PY
```

Do not use the generic 300 GB ground-up size. A historic object backup can be far larger. Record the
selected mode, selected bytes/count, calculated capacity, and RunPod console quote in
`$LOCAL_RESTORE_CONTROL/restore-decision.txt`.

## Stage 3 — create an empty replacement RunPod volume

In the RunPod browser:

1. Add enough account balance for the calculated storage and later validation Pod.
2. Open **Storage → New Network Volume**.
3. Name it `hjepa-vwm-local-restore-YYYYMMDD`.
4. Choose `US-MO-1` if available; otherwise choose a current S3-supported datacenter and use its
   documented endpoint/region below.
5. Enter the calculated capacity; volumes can grow but not shrink.
6. Review the live recurring price and create the volume.
7. Record the new volume ID, datacenter, signing region, endpoint, size, and quote. The ID will not
   be `4hzrwzk8ja`.
8. In **Settings → S3 API Keys**, create `temporary-local-restore-YYYYMMDD` and store its one-time
   secret in a password manager.

Set the exact new values on the Mac:

```bash
export NEW_RUNPOD_VOLUME_ID="REPLACE_WITH_NEW_VOLUME_ID"
export NEW_RUNPOD_REGION="us-mo-1"
export NEW_RUNPOD_ENDPOINT="https://s3api-us-mo-1.runpod.io"
```

Use the exact documented values for another datacenter; never mix an ID from one datacenter with
another endpoint.

## Stage 4 — configure the destination and persistent Mac session

Confirm rclone and install tmux if needed:

```bash
command -v rclone || brew install rclone
command -v tmux || brew install tmux
rclone version
```

Open a second Terminal and leave this running until Stage 7 finishes:

```bash
caffeinate -dimsu
```

In the first Terminal, start tmux:

```bash
tmux new -s hjepa-local-restore
```

Inside tmux, use in-memory configuration and hidden prompts:

```bash
export RCLONE_CONFIG=/dev/null
read -r -p "New RunPod S3 access key (user_...): " NEW_RUNPOD_S3_ACCESS_KEY
read -r -s -p "New RunPod S3 secret (rps_...): " NEW_RUNPOD_S3_SECRET_KEY; printf '\n'
```

Configure the new destination:

```bash
export RCLONE_CONFIG_NEWRUNPOD_TYPE=s3
export RCLONE_CONFIG_NEWRUNPOD_PROVIDER=Other
export RCLONE_CONFIG_NEWRUNPOD_ENV_AUTH=false
export RCLONE_CONFIG_NEWRUNPOD_ACCESS_KEY_ID="$NEW_RUNPOD_S3_ACCESS_KEY"
export RCLONE_CONFIG_NEWRUNPOD_SECRET_ACCESS_KEY="$NEW_RUNPOD_S3_SECRET_KEY"
export RCLONE_CONFIG_NEWRUNPOD_REGION="$NEW_RUNPOD_REGION"
export RCLONE_CONFIG_NEWRUNPOD_ENDPOINT="$NEW_RUNPOD_ENDPOINT"
export RCLONE_CONFIG_NEWRUNPOD_FORCE_PATH_STYLE=true
export RCLONE_CONFIG_NEWRUNPOD_NO_CHECK_BUCKET=true
```

Prove the remote exists and the new target is genuinely empty:

```bash
rclone listremotes
rclone lsf "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --max-depth 1
```

Expected: `newrunpod:` is listed and the second command succeeds with no paths. If anything exists,
stop and verify the ID. Account for/remove only test objects personally created on this brand-new
target, or create another empty volume; integrity checks require no destination extras.

## Stage 5A — recommended POSIX-aware upload

Define the exact filtered copy:

```bash
restore_local_posix_aware() { rclone copy "$LOCAL_VOLUME_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --s3-upload-concurrency 2 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$LOCAL_RESTORE_CONTROL/copy-posix-aware.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Run to two consecutive zero passes:

```bash
restore_local_posix_aware
printf 'last rclone exit status: %s\n' "$?"
```

Rerun the identical function until zero, then once more. Run a path/size check with identical
filters:

```bash
rclone check "$LOCAL_VOLUME_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' --size-only --checkers 8 --combined "$LOCAL_RESTORE_CONTROL/check-posix-aware-size.txt" --log-file "$LOCAL_RESTORE_CONTROL/check-posix-aware-size.log" --log-level INFO
```

Run the full-byte check with the same filters:

```bash
rclone check "$LOCAL_VOLUME_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' --download --checkers 4 --combined "$LOCAL_RESTORE_CONTROL/check-posix-aware-download.txt" --log-file "$LOCAL_RESTORE_CONTROL/check-posix-aware-download.log" --log-level INFO --stats 30s --stats-one-line -P
```

Both checks must exit zero. Continue at Stage 6.

## Stage 5B — literal object upload alternative

Use this instead of 5A only if the new capacity covers every byte plus headroom and regular files
at old link-view paths are acceptable.

```bash
restore_local_all() { rclone copy "$LOCAL_VOLUME_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --s3-upload-concurrency 2 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$LOCAL_RESTORE_CONTROL/copy-all.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Run to two consecutive zero passes:

```bash
restore_local_all
printf 'last rclone exit status: %s\n' "$?"
```

Run the full-byte check before adding control files:

```bash
rclone check "$LOCAL_VOLUME_DIR" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --download --checkers 4 --combined "$LOCAL_RESTORE_CONTROL/check-all-download.txt" --log-file "$LOCAL_RESTORE_CONTROL/check-all-download.log" --log-level INFO --stats 30s --stats-one-line -P
```

Require exit zero.

## Stage 6 — copy manifests and new restore evidence separately

Only after the selected data check exits zero, copy old rescue manifests to a separate root:

```bash
rclone copy "$LOCAL_RESCUE_ROOT/recovery-manifests" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-manifests" --size-only --transfers 2 --checkers 4 --retries 20 --low-level-retries 50 -P
```

Hash the new local restore reports:

```bash
(cd "$LOCAL_RESTORE_CONTROL" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | xargs -0 shasum -a 256 > SHA256SUMS)
```

Copy them under a unique control prefix, not into the restored data root:

```bash
rclone copy "$LOCAL_RESTORE_CONTROL" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-manifests/local-restore-${LOCAL_RESTORE_ID}" --size-only --transfers 2 --checkers 4 -P
```

Unset secrets and revoke the temporary key in RunPod Settings:

```bash
unset NEW_RUNPOD_S3_ACCESS_KEY NEW_RUNPOD_S3_SECRET_KEY RCLONE_CONFIG_NEWRUNPOD_ACCESS_KEY_ID RCLONE_CONFIG_NEWRUNPOD_SECRET_ACCESS_KEY
```

Stop the separate `caffeinate` with `Ctrl-C`. Do not erase or repurpose the rescue disk.

## Stage 7 — reconstruct filesystem topology on a Pod

Continue at
[`02_RESTORE_A_VERIFIED_AWS_BACKUP.md` Stage 7](02_RESTORE_A_VERIFIED_AWS_BACKUP.md#stage-7--deploy-a-pod-with-the-recovered-volume).
AWS-specific Stages 0–6 in that file do not apply; the local upload above replaced them.

On the Pod:

1. verify `/workspace` is the new mount;
2. reconstruct recorded directories from `source-directory-paths.csv`;
3. for POSIX-aware mode, reconstruct the six symlink views from the exact source inventory or
   preserved official/tiny manifests;
4. restore tracked Git modes and inspect repository identity;
5. run every gate in `07_VALIDATE_EXPERIMENT_READINESS.md`;
6. keep the external disk untouched until the new volume passes full science validation and a
   second independent backup exists.

Literal mode skips symlink creation for paths restored as valid regular videos. That can be
scientifically equivalent at the dataset-relative path/bytes/frame level, but it is not an exact
physical POSIX reconstruction.
